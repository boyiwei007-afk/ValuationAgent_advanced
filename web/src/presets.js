// Curated starting points for the setup UI. Every field remains editable so
// private gateways, newer model IDs, and companies outside this list work too.
export const MODEL_PRESETS = [
  {
    value: 'deepseek-flash',
    label: 'DeepSeek Flash',
    provider: 'openai_compatible',
    base_url: 'https://api.deepseek.com',
  },
  {
    value: 'deepseek-v4-pro',
    label: 'DeepSeek V4 Pro',
    provider: 'openai_compatible',
    base_url: 'https://api.deepseek.com',
  },
  {
    value: 'gpt-4o-mini',
    label: 'OpenAI GPT-4o mini',
    provider: 'openai',
    base_url: 'https://api.openai.com/v1',
  },
]
