import json
import re
import uuid


NAME = r"[A-Za-z_][A-Za-z0-9_.-]{0,127}"
CALL = re.compile(rf"<tool_call>\s*<function=({NAME})>(.*?)</function>\s*</tool_call>", re.S)
PARAMETER = re.compile(rf"<parameter=({NAME})>(.*?)</parameter>", re.S)
DELIMITER = re.compile(r"</?(?:tool_call|function|parameter)(?:[=>\s]|$)")


class ParameterSchemaError(ValueError):
    def __init__(self, name, variants):
        allowed = sorted({key for variant in variants for key in variant.get("properties", {})})
        self.public_reason = f"工具{name}的顶层参数须匹配同一schema分支；可用参数：" + ",".join(allowed[:32])
        super().__init__(self.public_reason)


def strict_json(text):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def reject(value):
        raise ValueError("non-finite JSON constant")

    value = json.loads(text, object_pairs_hook=unique, parse_constant=reject)
    json.dumps(value, allow_nan=False)
    return value


def parameter_types(schema, root, visited=()):
    reference = schema.get("$ref")
    if reference:
        if not reference.startswith("#/$defs/") or reference in visited:
            raise ValueError("unsupported schema reference")
        target = root.get("$defs", {}).get(reference.removeprefix("#/$defs/"))
        if not isinstance(target, dict):
            raise ValueError("missing schema reference")
        return parameter_types(target, root, (*visited, reference))
    kinds = schema.get("type", [])
    result = {kinds} if isinstance(kinds, str) else set(kinds)
    for branch in schema.get("anyOf", []) + schema.get("oneOf", []):
        result.update(parameter_types(branch, root, visited))
    return result


def decode_parameter(raw, schema, root):
    if DELIMITER.search(raw):
        raise ValueError("nested or duplicated protocol delimiter")
    raw = raw.removeprefix("\r\n").removeprefix("\n")
    raw = raw.removesuffix("\r\n").removesuffix("\n")
    kinds = parameter_types(schema, root)
    literal = raw.strip().lower()
    if "null" in kinds and literal in {"null", "none"}:
        return None
    if "string" in kinds:
        if raw.strip().startswith('"'):
            value = strict_json(raw)
            if not isinstance(value, str):
                raise ValueError("invalid string parameter")
            return value
        return raw
    if "boolean" in kinds:
        if literal not in {"true", "false"}:
            raise ValueError("invalid boolean parameter")
        return literal == "true"
    return strict_json(raw)


def object_variants(schema, root, visited=()):
    reference = schema.get("$ref")
    if reference:
        if not reference.startswith("#/$defs/") or reference in visited:
            raise ValueError("unsupported schema reference")
        target = root.get("$defs", {}).get(reference.removeprefix("#/$defs/"))
        if not isinstance(target, dict):
            raise ValueError("missing schema reference")
        return object_variants(target, root, (*visited, reference))
    branches = schema.get("anyOf") or schema.get("oneOf")
    if not branches:
        return [schema]
    variants = []
    for branch in branches:
        for nested in object_variants(branch, root, visited):
            variants.append({"properties": {**schema.get("properties", {}), **nested.get("properties", {})},
                "required": list(dict.fromkeys([*schema.get("required", []), *nested.get("required", [])]))})
            if len(variants) > 32:
                raise ValueError("too many schema branches")
    return variants


def decode_arguments(raw_arguments, schema, name):
    variants = object_variants(schema, schema)
    matches = []
    for variant in variants:
        properties = variant.get("properties", {})
        if not set(raw_arguments) <= set(properties) or not set(variant.get("required", [])) <= set(raw_arguments):
            continue
        try:
            arguments = {key: decode_parameter(raw, properties[key], schema) for key, raw in raw_arguments.items()}
        except (ValueError, TypeError, KeyError):
            continue
        if any("const" in properties[key] and value != properties[key]["const"]
               or "enum" in properties[key] and value not in properties[key]["enum"] for key, value in arguments.items()):
            continue
        matches.append(arguments)
    if not matches or schema.get("oneOf") and len(matches) != 1:
        raise ParameterSchemaError(name, variants)
    encoded = {json.dumps(arguments, ensure_ascii=False, sort_keys=True, allow_nan=False) for arguments in matches}
    if len(encoded) != 1:
        raise ValueError("ambiguous parameter types")
    return matches[0]


def normalize_qwen_call(message, tools):
    content = message.get("content")
    if not isinstance(content, str) or len(content) > 192000:
        raise ValueError("invalid response size or type")
    start = content.find("<tool_call>")
    if start < 0 or "<" in content[:start] or "```" in content[:start]:
        raise ValueError("expected complete tool calls")
    remaining_calls = content[start:].strip()
    functions = {item["function"]["name"]: item["function"] for item in tools if item.get("type") == "function"}
    calls = []
    while remaining_calls:
        match = CALL.match(remaining_calls)
        if match is None or len(calls) >= 6:
            raise ValueError("incomplete calls, trailing text or too many calls")
        name, remaining = match.groups()
        if name not in functions:
            raise ValueError("unregistered function")
        schema = functions[name]["parameters"]
        raw_arguments = {}
        while remaining.strip():
            parameter = PARAMETER.match(remaining.lstrip())
            if parameter is None:
                raise ValueError("malformed parameter block")
            key, raw = parameter.groups()
            if key in raw_arguments:
                raise ValueError("unregistered or duplicated parameter")
            raw_arguments[key] = raw
            remaining = remaining.lstrip()[parameter.end():]
        arguments = decode_arguments(raw_arguments, schema, name)
        encoded = json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        if len(encoded) > 32000:
            raise ValueError("arguments too large")
        calls.append({"id": "call_" + uuid.uuid4().hex, "type": "function", "function": {"name": name, "arguments": encoded}})
        remaining_calls = remaining_calls[match.end():].strip()
    result = {**message, "content": None, "tool_calls": calls}
    result.pop("reasoning_content", None)
    return result
