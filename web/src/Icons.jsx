const paths = {
  trash: 'M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7',
  plus: 'M12 5v14M5 12h14', arrow: 'M5 12h14m-6-6 6 6-6 6', send: 'm5 12 14-7-5 14-2-7-7 0Zm7 0 7-7',
  chat: 'M20 11a8 8 0 0 1-8 8H8l-5 3 1.5-6A8 8 0 1 1 20 11Z',
  chart: 'M4 4v16h16M8 15v-4m4 4V7m4 8v-6', file: 'M14 3H5v18h14V8l-5-5Zm0 0v5h5M8 12h8m-8 4h5',
  settings: 'M4 7h16M4 17h16M8 4v6m8 4v6', link: 'm10 13 4-4m-6 6-2 2a3 3 0 0 1-4-4l4-4a3 3 0 0 1 4 0m4 0 2-2a3 3 0 0 0-4-4l-4 4a3 3 0 0 0 0 4',
  check: 'm5 12 4 4L19 6', close: 'm6 6 12 12M6 18 18 6', chevron: 'm9 5 7 7-7 7',
  clock: 'M12 8v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
  play: 'm8 5 11 7-11 7V5Z', download: 'M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5',
  tool: 'm14 6 4 4 3-3a7 7 0 0 1-9 9l-6 6-4-4 6-6a7 7 0 0 1 9-9l-3 3Z',
  upload: 'M12 16V3m-5 5 5-5 5 5M4 16v5h16v-5', globe: 'M2 12h20M12 2c6 6 6 14 0 20-6-6-6-14 0-20Zm10 10a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
  layers: 'm12 3 10 6-10 6L2 9l10-6Zm-9 11 9 6 9-6', refresh: 'M20 7v6h-6M4 17v-6h6M5 7a8 8 0 0 1 13-2l2 2M4 17l2 2a8 8 0 0 0 13-2',
  alert: 'm12 3 10 18H2L12 3Zm0 6v5m0 3v.2', menu: 'M4 6h16M4 12h16M4 18h16',
  spark: 'm12 3 3 6 6 3-6 3-3 6-3-6-6-3 6-3 3-6Z', shield: 'm12 2 8 4v6c0 5-8 10-8 10S4 17 4 12V6l8-4Zm-4 10 3 3 5-6',
  code: 'm8 5-6 7 6 7m8-14 6 7-6 7m-3-16-2 18', folder: 'M3 5h6l2 3h10v12H3V5Z',
}
export default function Icon({ name, size = 20, ...props }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}><path d={paths[name] || paths.spark}/></svg>
}
