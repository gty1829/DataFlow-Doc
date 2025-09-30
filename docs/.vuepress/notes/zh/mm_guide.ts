import type { ThemeNote } from 'vuepress-theme-plume'
import { defineNoteConfig } from 'vuepress-theme-plume'

export const MMGuide: ThemeNote = defineNoteConfig({
    dir: 'mm_guide',
    link: '/mm_guide/',
    sidebar: [
        {
            text: 'Basic Info',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'basicinfo',
            items: [
                'intro',
                'framework',
            ],
        },
        {
            text: 'Start with Dataflow',
            collapsed: false,
            icon: 'carbon:idea',
            prefix: 'quickstart',
            items: [
                'install',
                'quickstart',
                'image_generation',
                'audio_caption',
                'whisper_asr',
                'video_caption',
                'audio_asr_pipeline',
            ],
        },
    ]
})
