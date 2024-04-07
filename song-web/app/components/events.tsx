'use client'

export enum EventType {
    Empire,
    Year,
    Artist,
    Ou, Yan, Liu, Zhao, Others,
    Overview,
    Content,
};

export type TimelineEvent = {
    name: string;
    year: number;
    ty: EventType;
    src: string;
    desc: string;
};

export type TimeLineEvents = {
    events: TimelineEvent[];
    span: [number, number];
};


export function LoadEvents() {
    let events: TimeLineEvents[] = [{
        events: [
            { name: "宋太祖--赵匡胤", year: 960, ty: EventType.Empire, src: "赵匡胤.png", desc: "宋太祖赵匡胤" },
            { name: "建隆", year: 960, ty: EventType.Year, src: "", desc: "建隆" },
            { name: "乾徳", year: 963, ty: EventType.Year, src: "", desc: "乾徳" },
            { name: "开宝", year: 968, ty: EventType.Year, src: "", desc: "开宝" },
            { name: "开宝藏", year: 971, ty: EventType.Content, src: "《开宝藏》.jpg", desc: "沿袭「唐楷」——《开宝藏》大藏经“自写本”转向“刻本”" },
            { name: "炽盛光佛顶大威德销灾吉祥陀罗尼经", year: 972, ty: EventType.Content, src: "《炽盛光佛顶大威德销灾吉祥陀罗尼经》.jpg", desc: "近元赵孟頫/有很大的相似性 ——《炽盛光佛顶大威德销灾吉祥陀罗尼经》" },
            { name: "宋太宗--赵炅", year: 977, ty: EventType.Empire, src: "赵炅.png", desc: "宋太宗赵炅" },
            { name: "太平兴国", year: 977, ty: EventType.Year, src: "", desc: "太平兴国" },
            { name: "雍熙", year: 984, ty: EventType.Year, src: "", desc: "雍熙" },
            { name: "文殊菩萨像", year: 984, ty: EventType.Content, src: "《文殊菩萨像》.jpg", desc: "大都仿欧阳询——《文殊菩萨像》/《弥勒菩萨像》/《普贤菩萨像》" },
            { name: "佛说北斗七星经", year: 986, ty: EventType.Content, src: "《佛说北斗七星经》.jpg", desc: "北魏写经体——《佛说北斗七星经》" },
            { name: "端拱", year: 988, ty: EventType.Year, src: "", desc: "端拱" },
            { name: "淳化", year: 990, ty: EventType.Year, src: "", desc: "淳化" },
            { name: "大方广佛华严经", year: 990, ty: EventType.Content, src: "《大方广佛华严经》.jpg", desc: "北魏写经体遗风——《大方广佛华严经》" },
            { name: "至到", year: 995, ty: EventType.Year, src: "", desc: "至到" },
            { name: "咸平", year: 998, ty: EventType.Year, src: "", desc: "咸平" },
            { name: "宋真宗——赵恒", year: 998, ty: EventType.Empire, src: "赵恒.png", desc: "宋真宗赵恒" },
            { name: "景德", year: 1004, ty: EventType.Year, src: "", desc: "景德" },
            { name: "大中祥符", year: 1008, ty: EventType.Year, src: "", desc: "大中祥符" },
            { name: "天禧", year: 1017, ty: EventType.Year, src: "", desc: "天禧" },
        ],
        span: [960, 1020]
    },
    {
        events: [
            { name: "乾兴", year: 1022, ty: EventType.Year, src: "", desc: "乾兴" },
            { name: "天圣", year: 1023, ty: EventType.Year, src: "", desc: "天圣" },
            { name: "宋仁宗——赵祯", year: 1023, ty: EventType.Empire, src: "赵祯.png", desc: "宋仁宗赵祯" },
            { name: "明道", year: 1032, ty: EventType.Year, src: "", desc: "明道" },
            { name: "景祐", year: 1034, ty: EventType.Year, src: "", desc: "景祐" },
            { name: "宝元", year: 1038, ty: EventType.Year, src: "", desc: "宝元" },
            { name: "康定", year: 1040, ty: EventType.Year, src: "", desc: "康定" },
            { name: "庆历", year: 1041, ty: EventType.Year, src: "", desc: "庆历" },
            { name: "皇祐", year: 1049, ty: EventType.Year, src: "", desc: "皇祐" },
            { name: "妙法莲华经", year: 1051, ty: EventType.Content, src: "《妙法莲华经》七卷.jpg", desc: "类欧阳率更" },
            { name: "至和", year: 1054, ty: EventType.Year, src: "", desc: "至和" },
            { name: "嘉祐", year: 1056, ty: EventType.Year, src: "", desc: "嘉祐" },
            { name: "佛顶心观世音菩萨大陀罗尼经", year: 1063, ty: EventType.Content, src: "《佛顶心观世音菩萨大陀罗尼经》.jpg", desc: "经文字体取法颜鲁公《多宝塔》" },
            { name: "治平", year: 1064, ty: EventType.Year, src: "", desc: "治平" },
            { name: "宋英宗——赵曙", year: 1064, ty: EventType.Empire, src: "赵曙.png", desc: "宋英宗赵曙" },
            { name: "熙宁", year: 1068, ty: EventType.Year, src: "", desc: "熙宁" },
            { name: "宋神宗——赵顼", year: 1068, ty: EventType.Empire, src: "赵顼.png", desc: "宋神宗赵顼" },
            { name: "元丰", year: 1078, ty: EventType.Year, src: "", desc: "元丰" },
        ],
        span: [1022, 1080]
    },
    ]

    return events;
}