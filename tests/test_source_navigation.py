import io

import httpx
import pytest
from pypdf import PdfWriter
from pypdf.annotations import Link

from valuationagent.application.source_navigation import SourceLinks, FollowSourceLink, list_source_links, follow_source_link
from valuationagent.schemas.research import DocumentSummary
from test_multisource_extraction import attach, runtime_at


def html_source(runtime, body):
    blocks = attach(runtime, "index.html", body)
    document = runtime.session.documents[-1]
    document.source_url = "https://publisher.example.test/ir/index.html"
    return document.file_id


def test_html_links_resolve_base_decode_entities_and_exclude_unsafe_actions(tmp_path):
    runtime = runtime_at(tmp_path)
    file_id = html_source(runtime, '<base href="/reports/"><a href="annual.html?year=2025&amp;type=full">年度报告</a>'
                                  '<a href="annual.html?year=2025&amp;type=full#p3">重复</a>'
                                  '<script>fetch("https://attacker.example.test")</script>'
                                  '<a href="javascript:alert(1)">执行</a><a href="file:///C:/secret">文件</a>'
                                  '<a href="https://127.0.0.1/private">内网</a><a href="https://name:pass@example.test">认证</a>')
    result = list_source_links(runtime.service.store, runtime.session, SourceLinks(file_id=file_id))
    assert result["total_matches"] == 1
    link = result["links"][0]
    assert link["url"] == "https://publisher.example.test/reports/annual.html?year=2025&type=full"
    assert "年度报告" in link["text"]
    assert result["untrusted_source_data"]
    assert not runtime.session.facts


def test_follow_saves_provenance_without_inheriting_parent_grade_or_publication_date(tmp_path):
    runtime = runtime_at(tmp_path)
    file_id = html_source(runtime, '<p>发布日期：2026-04-03</p><a href="/child">详细财报</a>')
    calls = []

    def transport(request):
        calls.append(str(request.url))
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<h1>子文件</h1><p>本文件未提供发布日期</p>")

    runtime.service.remote_transport = httpx.MockTransport(transport)
    row = list_source_links(runtime.service.store, runtime.session, SourceLinks(file_id=file_id))["links"][0]
    args = FollowSourceLink(**row["follow_arguments"])
    result = follow_source_link(runtime, args)
    document = runtime.session.documents[-1]
    assert document.authority_tier == "C" and document.acquisition_ref == args.link_id
    blocks = runtime.service.store.research_blocks(runtime.session.session_id, result["file_id"])
    assert blocks[0]["location"]["published_at"] is None
    assert blocks[0]["location"]["parent_source_file_id"] == file_id
    assert blocks[0]["location"]["parent_search_file_id"] is None
    assert blocks[0]["location"]["source_link_id"] == args.link_id
    assert result["parent_source_file_id"] == file_id
    assert follow_source_link(runtime, args)["status"] == "already_fetched"
    assert len(calls) == 1


def test_link_id_is_bound_to_immutable_original_not_model_supplied_url(tmp_path):
    runtime = runtime_at(tmp_path)
    original = html_source(runtime, '<a href="https://publisher.example.test/report">报告</a>')
    another = html_source(runtime, '<p>different bytes</p><a href="https://publisher.example.test/report">同一网址</a>')
    link = list_source_links(runtime.service.store, runtime.session, SourceLinks(file_id=original))["links"][0]
    with pytest.raises(ValueError, match="SOURCE_LINK_NOT_FOUND"):
        follow_source_link(runtime, FollowSourceLink(file_id=another, link_id=link["link_id"]))
    assert len(runtime.session.documents) == 2


def test_upload_only_disallows_following_and_redirect_to_loopback_is_blocked(tmp_path):
    runtime = runtime_at(tmp_path)
    file_id = html_source(runtime, '<a href="https://publisher.example.test/report">报告</a>')
    args = FollowSourceLink(**list_source_links(runtime.service.store, runtime.session, SourceLinks(file_id=file_id))["links"][0]["follow_arguments"])
    calls = []

    def transport(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://127.0.0.1/private"})

    runtime.service.remote_transport = httpx.MockTransport(transport)
    runtime.session.data_source_preference = "upload"
    with pytest.raises(ValueError, match="NETWORK_OUT_OF_SCOPE"):
        follow_source_link(runtime, args)
    assert calls == []
    runtime.session.data_source_preference = "web"
    with pytest.raises(ValueError, match="私有|回环"):
        follow_source_link(runtime, args)
    assert len(calls) == 1 and len(runtime.session.documents) == 1
    with pytest.raises(ValueError, match="SOURCE_LINK_REPEATED"):
        follow_source_link(runtime, args)


def test_pagination_can_follow_a_link_beyond_first_display_window(tmp_path):
    runtime = runtime_at(tmp_path)
    file_id = html_source(runtime, "".join(f'<a href="/reports/{number}">报告{number}</a>' for number in range(60)))
    runtime.service.remote_transport = httpx.MockTransport(lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, text="source"))
    packet = list_source_links(runtime.service.store, runtime.session, SourceLinks(file_id=file_id, offset=50, limit=2))
    assert packet["total_matches"] == 60 and packet["next_offset"] == 52
    result = follow_source_link(runtime, FollowSourceLink(**packet["links"][0]["follow_arguments"]))
    assert result["source_url"].endswith("/50")


def test_pdf_annotations_use_real_page_and_do_not_execute_actions(tmp_path):
    runtime = runtime_at(tmp_path)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    writer.add_annotation(page_number=1, annotation=Link(rect=(0, 0, 50, 50), url="https://publisher.example.test/original"))
    stream = io.BytesIO()
    writer.write(stream)
    meta = runtime.service.store.save_upload("links.pdf", "evidence", "application/pdf", stream.getvalue())
    runtime.session.documents.append(DocumentSummary(file_id=meta["file_id"], name="links.pdf", role="evidence",
        block_count=0, sha256=meta["sha256"], provenance_type="user_upload", authority_tier="B"))
    first = list_source_links(runtime.service.store, runtime.session, SourceLinks(file_id=meta["file_id"], page_count=1))
    assert first["next_page"] == 2 and first["links"] == []
    second = list_source_links(runtime.service.store, runtime.session, SourceLinks(file_id=meta["file_id"], start_page=2))
    assert second["links"][0]["page"] == 2
    assert second["links"][0]["follow_arguments"]["page"] == 2


def test_link_network_budget_does_not_reset_or_create_sources(tmp_path):
    runtime = runtime_at(tmp_path)
    file_id = html_source(runtime, '<a href="/report">报告</a>')
    args = FollowSourceLink(**list_source_links(runtime.service.store, runtime.session, SourceLinks(file_id=file_id))["links"][0]["follow_arguments"])
    runtime.followed_links = {f"attempt_{index}" for index in range(8)}
    with pytest.raises(ValueError, match="SOURCE_LINK_BUDGET"):
        follow_source_link(runtime, args)
    assert len(runtime.session.documents) == 1
