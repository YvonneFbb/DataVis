'use client'


export type EmpireEvent = {
    name: string;
    year: number;
    src: string;
};

export type YearEvent = {
    name: string;
    year: number
}

export type ArtistEvent = {
    name: string;
    year: number;
    end: number;
    src: string;
    desc: string;
}

export type ContentEvent = {
    name: string;
    src: string;
    title: string;
    desc: string;
}

export type ContentsEvent = {
    contents: ContentEvent[];
    year: number;
    len: number;
}

export enum EventType {
    Empire,
    Year,
    Artist,
    Ou, Yan, Liu, Zhao, Others,
    Overview,
    Contents,
};

export type TimelineEvent = {
    event: EmpireEvent | YearEvent | ArtistEvent | ContentsEvent;
    type: EventType;
}

// export type TimelineEvent = {
//     name: string;
//     year: number;
//     ty: EventType;
//     src: string;
//     title: string;
//     desc: string;
// };

export type TimeLineEvents = {
    events: TimelineEvent[];
    span: [number, number];
    desc: string,
};


export function LoadEvents() {
    let events: TimeLineEvents[] = [
        // 0
        {
            events: [
                { event: { name: "欧阳询", year: 557, end: 641, src: "欧阳询.png", desc: "欧阳询精通书法，于平正中见险绝，号为「欧体」。楷书代表作包括：《九成宫醴泉铭》，《皇甫诞碑》，《化度寺碑》。所写《化度寺邑禅师舍利塔铭》、《虞恭公温彦博碑》、《皇甫诞碑》被称为「唐人楷书第一」。" }, type: EventType.Artist },
                { event: { year: 586, len: 1, contents: [{ name: "《龙藏寺碑》", src: "《龙藏寺碑》.jpg", title: "<span class='ou-char'>欧</span>阳询", desc: "魏碑体到唐碑体发展的过渡时期" }] }, type: EventType.Contents },
            ],
            span: [420, 589],
            desc: "南北朝"
        },
        // 1
        {
            events: [
                { event: { year: 586, len: 1, contents: [{ name: "《龙藏寺碑》", src: "《龙藏寺碑》.jpg", title: "<span class='ou-char'>欧</span>阳询", desc: "魏碑体到唐碑体发展的过渡时期" }] }, type: EventType.Contents },
                { event: { name: "褚遂良", year: 596, end: 658, src: "褚遂良.png", desc: "褚体合欧虞两家之长，能自成一派，完成意与法的高度协调，“真正开启‘唐楷’门户，堪称隋唐楷书过度的桥梁”，最终推动了唐代尚法书风的形成。" }, type: EventType.Artist },
                { event: { year: 597, len: 1, contents: [{ name: "《董美人墓志》", src: "《董美人墓志》.jpg", title: "小楷，隋志小楷第一", desc: "上承北魏书体，下开唐代书风，是南北朝到唐之间的津梁，开启了唐代小楷的先河，唐初虞、欧等皆承于此。" }] }, type: EventType.Contents },

            ],
            span: [581, 618],
            desc: "隋朝"
        },
        // 2
        {
            events: [
                { event: { year: 632, len: 1, contents: [{ name: "《九成宫醴泉铭》", src: "《九成宫醴泉铭》.jpg", title: "<span class='ou-char'>欧</span>阳询", desc: "楷书之极则，正书第一" }] }, type: EventType.Contents },
                { event: { year: 653, len: 1, contents: [{ name: "《雁塔圣教序》", src: "《雁塔圣教序》.jpg", title: "褚遂良", desc: "引领大唐楷书新格最能代表褚遂良楷书风格的作品" }] }, type: EventType.Contents },
                { event: { name: "颜真卿", year: 709, end: 784, src: "颜真卿.png", desc: "楷书艺术典范，彻底摆脱了初唐的风范，创造了新的时代书风，具有划时代的贡献。“生新法于占意之外，熔铸万象隐括众生，转移风尚。”" }, type: EventType.Artist },
                { event: { year: 752, len: 1, contents: [{ name: "《多宝塔感应碑》", src: "《多宝塔感应碑》.jpg", title: "<span class='yan-char'>颜</span>真卿", desc: "颜真卿早期的得意之作，书写严谨真实，承接了当时唐朝书法家褚遂良的风格。" }] }, type: EventType.Contents },
                { event: { year: 771, len: 1, contents: [{ name: "《麻姑仙坛记》", src: "《麻姑仙坛记》.jpg", title: "<span class='yan-char'>颜</span>真卿", desc: "颜真卿楷书的代表作，此时颜真卿楷书风格已基本完善。" }] }, type: EventType.Contents },
                { event: { name: "柳公权", year: 778, end: 865, src: "柳公权.png", desc: "最后完成楷书定型的杰出书家，唐代书法达到了艺术的顶峰。由颜真卿出，合以欧阳询能独立门户。" }, type: EventType.Artist },
                { event: { year: 779, len: 1, contents: [{ name: "《颜勤礼碑》", src: "《颜勤礼碑》.jpg", title: "<span class='yan-char'>颜</span>真卿", desc: "颜真卿晚年楷书的代表作，其书法艺术已臻成熟之境，代表盛唐审美风尚。" }] }, type: EventType.Contents },
                { event: { year: 824, len: 1, contents: [{ name: "《金刚经刻石》", src: "《金刚经刻石》.jpg", title: "<span class='liu-char'>柳</span>公权", desc: "柳书早期代表作，“柳骨”于此可初识" }] }, type: EventType.Contents },
                { event: { year: 837, len: 1, contents: [{ name: "《冯宿碑》", src: "《冯宿碑》.jpg", title: "<span class='liu-char'>柳</span>公权", desc: "蝉蜕前夕的作品预示一种更为精炼的“柳体”即将孕育而出。" }] }, type: EventType.Contents },
                { event: { year: 841, len: 1, contents: [{ name: "《玄秘塔碑》", src: "《玄秘塔碑》.jpg", title: "<span class='liu-char'>柳</span>公权", desc: "柳公权书法创作生涯中的一座里程碑，标志着“柳体”书法的完全成熟。" }] }, type: EventType.Contents },

            ],
            span: [618, 907],
            desc: "唐朝"
        },
        // 3
        {
            events: [
                { event: { name: "宋太祖赵匡胤", year: 960, src: "赵匡胤.png" }, type: EventType.Empire },
                { event: { name: "建隆", year: 960 }, type: EventType.Year },
                { event: { name: "乾徳", year: 963 }, type: EventType.Year },
                { event: { name: "开宝", year: 968 }, type: EventType.Year },
                { event: { year: 971, len: 1, contents: [{ name: "《开宝藏》", src: "《开宝藏》.jpg", title: "第一部版刻大藏经，沿袭唐楷", desc: "《开宝藏》的问世具有非凡的意义。它标志大藏经完成了自写本向刻本的转变，也标志着版刻书法运用范畴进一步扩大，逐渐体现出版刻作品在短时间内可以“化身千亿”的优势。" }] }, type: EventType.Contents },
                { event: { year: 972, len: 1, contents: [{ name: "《炽盛光佛顶大威德销灾吉祥陀罗尼经》", src: "《炽盛光佛顶大威德销灾吉祥陀罗尼经》.jpg", title: "继承了唐代的标准写经风格", desc: "运笔带行书笔意，写刻精美，结体精严，点画飞动，有血有肉，转侧照人，《炽盛光佛顶大威德销灾吉祥陀罗尼经》字体的用笔、结体及使转与元赵孟頫书法有很大的相似性。" }] }, type: EventType.Contents },
                { event: { name: "宋太宗赵炅", year: 977, src: "赵炅.png" }, type: EventType.Empire },
                { event: { name: "太平兴国", year: 977 }, type: EventType.Year },
                { event: { name: "雍熙", year: 984 }, type: EventType.Year },
                { event: { year: 984, len: 1, contents: [{ name: "《文殊菩萨像》", src: "《文殊菩萨像》.jpg", title: "继承了唐代大都效仿<span class='ou-char'>欧<span>阳询的标准写经风格", desc: "《文殊菩萨像》险峻清劲，镌刻工整，书画同雕，具有典型意义。" }] }, type: EventType.Contents },
                { event: { year: 986, len: 1, contents: [{ name: "《佛说北斗七星经》", src: "《佛说北斗七星经》.jpg", title: "具有北魏写经体的体式", desc: "北宋雍熙三年刻本《佛说北斗七星经》卷轴装，上下单边卷尾有朱印六枚，字体为北魏写经体，写刻精湛。" }] }, type: EventType.Contents },
                { event: { name: "端拱", year: 988 }, type: EventType.Year },
                { event: { name: "淳化", year: 990 }, type: EventType.Year },
                { event: { year: 990, len: 1, contents: [{ name: "《大方广佛华严经》", src: "《大方广佛华严经》.jpg", title: "北魏写经体遗风", desc: "《大方广佛华严经》于北宋咸平年间杭州龙兴寺所刻，经折装。正文字体笔画方严，结字奇崛，有北魏写经体遗风" }] }, type: EventType.Contents },
                { event: { name: "至到", year: 995 }, type: EventType.Year },
                { event: { name: "咸平", year: 998 }, type: EventType.Year },
                { event: { name: "宋真宗赵恒", year: 998, src: "赵恒.png" }, type: EventType.Empire },
                { event: { name: "景德", year: 1004 }, type: EventType.Year },
                { event: { name: "大中祥符", year: 1008 }, type: EventType.Year },
                { event: { name: "天禧", year: 1017 }, type: EventType.Year },
            ],
            span: [960, 1020],
            desc: "北宋早期",
        },
        // 4
        {
            events: [
                { event: { name: "乾兴", year: 1022 }, type: EventType.Year },
                { event: { name: "天圣", year: 1023 }, type: EventType.Year },
                { event: { name: "宋仁宗赵祯", year: 998, src: "赵祯.png" }, type: EventType.Empire },
                { event: { name: "明道", year: 1032 }, type: EventType.Year },
                { event: { name: "景祐", year: 1034 }, type: EventType.Year },
                { event: { name: "宝元", year: 1038 }, type: EventType.Year },
                { event: { name: "康定", year: 1040 }, type: EventType.Year },
                { event: { name: "庆历", year: 1041 }, type: EventType.Year },
                { event: { name: "皇祐", year: 1049 }, type: EventType.Year },
                { event: { year: 1051, len: 1, contents: [{ name: "《妙法莲华经》", src: "《妙法莲华经》七卷.jpg", title: "类<span class='ou-char'>欧</span>阳率更", desc: "《妙法莲华经》七卷，北宋皇祐三年刊本。经折装，金粟山藏经纸刷印。上下单边。每版厘为五个半页，每半页五行，行十七字。字体类欧阳率更，笔法精严，刷印不苟。" }] }, type: EventType.Contents },
                { event: { name: "至和", year: 1054 }, type: EventType.Year },
                { event: { name: "嘉祐", year: 1056 }, type: EventType.Year },
                { event: { year: 1063, len: 1, contents: [{ name: "《佛顶心观世音菩萨大陀罗尼经》", src: "《佛顶心观世音菩萨大陀罗尼经》.jpg", title: "字体取法<span class='yan-char'>颜</span>鲁公", desc: "北宋嘉祐八年佛顶心观世音菩萨大陀罗尼经》，卷轴装。经文字体取法颜鲁公《多宝塔》。" }] }, type: EventType.Contents },
                { event: { name: "治平", year: 1064 }, type: EventType.Year },
                { event: { name: "宋英宗赵曙", year: 1064, src: "赵曙.png" }, type: EventType.Empire },
                { event: { name: "熙宁", year: 1068 }, type: EventType.Year },
                { event: { name: "宋神宗赵顼", year: 1068, src: "赵顼.png" }, type: EventType.Empire },
                { event: { name: "元丰", year: 1078 }, type: EventType.Year },
                {
                    event: {
                        year: 1080, contents: [
                            { name: "《崇宁藏》", src: "北宋福州崇宁藏本《一切经音义》.jpg", title: "<span class='ou-char'>欧</span><span class='liu-char'>柳</span>型", desc: "包含欧、颜、柳三种风格" },
                            { name: "《崇宁藏》", src: "北宋福州崇宁藏本《显扬圣教论》.jpg", title: "<span class='ou-char'>欧</span><span class='liu-char'>柳</span>型", desc: "包含欧、颜、柳三种风格" },
                            { name: "《崇宁藏》", src: "北宋福州崇宁藏本《说一切有部集异门足论》.jpg", title: "近<span class='ou-char'>欧</span>型", desc: "包含欧、颜、柳三种风格" },
                            { name: "《崇宁藏》", src: "北宋福州崇宁藏本《放光波若波罗蜜经》.jpg", title: "近<span class='yan-char'>颜</span>型，丰厚型", desc: "包含欧、颜、柳三种风格" },
                            { name: "《崇宁藏》", src: "北宋福州崇宁藏本《金刚顶瑜伽中略出念诵经》.jpg", title: "近<span class='yan-char'>颜</span>型，瘦弱型", desc: "包含欧、颜、柳三种风格" },
                        ],
                        len: 5,
                    }, type: EventType.Contents,
                },
            ],
            span: [1022, 1080],
            desc: "北宋中期",
        },
        // 5
        {
            events: [
                { event: { name: "元祐", year: 1086 }, type: EventType.Year },
                { event: { name: "宋哲宗赵煦", year: 1086, src: "赵煦.png" }, type: EventType.Empire },
                { event: { name: "绍圣", year: 1094 }, type: EventType.Year },
                { event: { name: "元符", year: 1098 }, type: EventType.Year },
                { event: { name: "建中靖国", year: 1101 }, type: EventType.Year },
                { event: { name: "宋徽宗赵佶", year: 1101, src: "赵佶.png" }, type: EventType.Empire },
                { event: { year: 1102, len: 1, contents: [{ name: "《妙法莲华经》", src: "《妙法莲华经》七卷.jpg", title: "楷法精妙，雅肖坡公", desc: "北宋崇宁刻本《妙法莲华经》，卷轴装，每行二十七字。上下单边。虽蝇头细楷，但镌刻工雅，颇见精神。字体扁平，笔画丰腴，普遍向右上方仰侧取势，为典型的苏轼楷书风格。" }] }, type: EventType.Contents },

                { event: { name: "大观", year: 1107 }, type: EventType.Year },
                { event: { name: "政和", year: 1111 }, type: EventType.Year },
                { event: { name: "重合", year: 1118 }, type: EventType.Year },
                { event: { name: "宣和", year: 1119 }, type: EventType.Year },
                { event: { name: "靖康", year: 1126 }, type: EventType.Year },
                { event: { name: "宋钦宗赵桓", year: 1126, src: "赵桓.png" }, type: EventType.Empire },
                {
                    event: {
                        year: 1126, len: 2, contents: [
                            { name: "《圆觉藏》", src: "《圆觉藏》.png", title: "颇具平原风采", desc: "北宋靖康元年刻大藏经《圆觉藏》，经折装。半页六行，行十七字。上下单边，千字文号为“宿”。上边框上方钤有“圆觉藏司自纸板”墨印。雄强圆厚，庄严雄浑。" },
                            { name: "《毗卢藏》", src: "《毗卢藏》.png", title: "意仿<span class='yan-char'>颜</span>鲁公", desc: "福州开元寺于宋徽宗政和二年开雕一部大藏经《毗卢藏》经折装。半页六行，行十七字。上下单边，千字文号为“列”，字体宽博雄浑。" }
                        ]
                    }, type: EventType.Contents
                },

            ],
            span: [1085, 1127],
            desc: "北宋晚期"
        },
        // 6
        {
            events: [
                { event: { name: "建炎", year: 1127 }, type: EventType.Year },
                { event: { name: "宋高宗赵构", year: 1127, src: "赵构.png" }, type: EventType.Empire },
                {
                    event: {
                        year: 1127, len: 3, contents: [
                            { name: "《长短经》", src: "《长短经》.png", title: "近<span class='ou-char'>欧</span>型", desc: "南宋初年杭州净戒院刻本《长短经》，字形瘦长，戈戟森严。看似平正，实则险劲。具备欧体字的大多特征。" },
                            { name: "《尚书正义》", src: "《尚书正义》.png", title: "近<span class='ou-char'>欧</span>型", desc: "字形瘦长，戈戟森严。看似平正，实则险劲。具备欧体字的大多特征。" },
                            { name: "《周易注疏》", src: "《周易注疏》.png", title: "近<span class='ou-char'>欧</span>型", desc: "字形瘦长，戈戟森严。看似平正，实则险劲。具备欧体字的大多特征。" }
                        ]
                    }, type: EventType.Contents
                },
                // TODO: short distance in timeline
                { event: { name: "绍兴", year: 1131 }, type: EventType.Year },
                { event: { year: 1132, len: 1, contents: [{ name: "《资治通鉴》", src: "《资治通鉴》.png", title: "（多半字体）<span class='ou-char'>欧</span><span class='yan-char'>颜</span>型", desc: "绍兴二至三年两浙东路茶盐司公使库刻本《资治通鉴》字体多半采用欧颜型。" }] }, type: EventType.Contents },
                { event: { year: 1139, len: 1, contents: [{ name: "《汉官仪》", src: "《汉官仪》.png", title: "近<span class='liu-char'>柳</span>型", desc: "绍兴九年临安府刻《汉官仪》字体爽利挺秀，骨力遒劲，具有典型柳体特征。传世南宋前期两浙地区刻本采用柳体者不多见，" }] }, type: EventType.Contents },
                { event: { year: 1140, len: 1, contents: [{ name: "《文粹》", src: "《文粹》.png", title: "书中楷书可分四种风格", desc: "宋绍兴九年临安府刻本《文粹》中的楷书风格其一，精劲如欧体者占大多数，为该书的主要基调；其二，宽博如颜体者；其三，方正如褚遂良《孟法师碑》者，与欧体之细长差别较大。此外，亦有写、刻不佳处，下刀迟疑，刀力软弱，字形歪曲，模糊不清，大概是覆刻北宋刻本的结果。可见，南宋绍兴刻本《文粹》至少有三位书手。" }] }, type: EventType.Contents },
                { event: { year: 1146, len: 1, contents: [{ name: "《事类赋》", src: "《事类赋》.jpg", title: "近<span class='ou-char'>欧</span>型", desc: "绍兴十六年两浙东路茶盐司刻本《事类赋》，半页八行，行十六至二十字不等。小字双行，行二十五至二十七字不等。白口，左右双边。书字体取法欧阳，字口清晰。" }] }, type: EventType.Contents },
                { event: { year: 1147, len: 1, contents: [{ name: "《古三坟书》", src: "《古三坟书》.png", title: "近<span class='yan-char'>颜</span>型", desc: "绍兴十七年婺州州学刻本《古三坟书》以颜字为基，又略加己意，形成一种别具面貌的颜字。在雕版风格上，跳出一般江浙版刻的方整传统，别树一帜，字体瘦劲。" }] }, type: EventType.Contents },
                { event: { year: 1148, len: 1, contents: [{ name: "《花间集》", src: "《花间集》.png", title: "<span class='yan-char'>颜</span>体", desc: "宋绍兴十八年建康郡斋刻《花间集》取法颜体。下刀软弱，笔画细瘦，使得原本雄壮的颜体，有女才貌，而无丈夫气。此书纸墨瑩洁，字体娟秀，在宋版书中别具风格。" }] }, type: EventType.Contents },
                { event: { year: 1152, len: 1, contents: [{ name: "《抱朴子》", src: "《抱朴子》.png", title: "颇有<span class='yan-char'>颜</span>鲁公风采", desc: "宋绍兴二十二年临安府荣六郎家刻《抱朴子》字体宽博，有颜鲁公风采。" }] }, type: EventType.Contents },
                { event: { name: "隆兴", year: 1131 }, type: EventType.Year },
                { event: { name: "宋孝宗赵昚", year: 1163, src: "赵昚.png" }, type: EventType.Empire },
            ],
            span: [1127, 1163],
            desc: "南宋早期"
        },
        // 7
        {
            events: [
                { event: { name: "乾道", year: 1165 }, type: EventType.Year },
                {
                    event: {
                        year: 1165, len: 2, contents: [
                            { name: "《洪氏集验方》五卷", src: "《洪氏集验方》五卷.jpg", title: "主要取法<span class='yan-char'>颜</span>平原和<span class='ou-char'>欧</span>阳率更", desc: "当涂姑孰郡斋乾道刊本《洪氏集验方》，字体妩媚，缺少浙本斩方肃穆之气。" },
                            { name: "《陈书》", src: "《陈书》.png", title: "扁方<span class='ou-char'>欧</span>体", desc: "南宋乾道年间出现扁方的欧体之后，这种字体在南宋中期被浙刻本广泛使用。著名的  “眉山七史”中的《陈书》体势扁瘦而笔画劲细。" },
                        ]
                    }, type: EventType.Contents
                },
                { event: { year: 1167, len: 1, contents: [{ name: "《论衡》", src: "《论衡》.jpg", title: "扁方<span class='ou-char'>欧</span>体", desc: "大约在南宋乾道年间，浙江地区出现一种将原本修长的欧体变为扁方字体，且字距加大，版面因此显得较为疏朗。这种字体在南宋中期得以广泛运用。宋乾道三年绍兴府刻本《论衡》为其代表。" }] }, type: EventType.Contents },
                { event: { year: 1169, len: 1, contents: [{ name: "《钜宋广韵》", src: "《钜宋广韵》.png", title: "有<span class='ou-char'>欧</span><span class='yan-char'>颜</span>之风", desc: "宋乾道五年建宁府黄三八郎刻本《钜宋广韵》字体清秀，笔画细瘦，粗细一致，结字有欧、颜之风，与早期浙本风格多有相近，气息也有些古朴。" }] }, type: EventType.Contents },
                { event: { year: 1172, len: 1, contents: [{ name: "《毗卢院施主忌晨记》", src: "《毗卢院施主忌晨记》.png", title: "<span class='yan-char'>颜</span>字", desc: "古代四川地区刻书通常被称作“蜀本”、“川本”等。唐宋时期，四川地区颜体较为流行。南宋乾道八年夹江《毗卢院施主忌晨记》笔画肥劲朴厚，结体宽博，颜体特征极其明显。" }] }, type: EventType.Contents },
                { event: { name: "淳熙", year: 1174 }, type: EventType.Year },
                { event: { year: 1175, len: 1, contents: [{ name: "《新定三礼图》", src: "《新定三礼图》.jpg", title: "<span class='ou-char'>欧</span><span class='yan-char'>颜</span>型", desc: "到了南宋中期，近欧型依然盛行，其他三种类型则较为少见。欧颜型仅见于宋淳熙二年镇江府学刻公文纸印本《新定三礼图》一书，惜笔画细瘦，俊美有余，古意不足。" }] }, type: EventType.Contents },
                { event: { year: 1178, len: 1, contents: [{ name: "《窦氏联珠集》", src: "《窦氏联珠集》.jpg", title: "略带褚遂良笔意", desc: "宋淳熙五年，略带褚遂良笔意，在宋代版刻楷书中别具一格。" }] }, type: EventType.Contents },
                { event: { name: "绍熙", year: 1190 }, type: EventType.Year },
                { event: { name: "宋光宗赵惇", year: 1190, src: "赵惇.png" }, type: EventType.Empire },
                { event: { year: 1193, len: 1, contents: [{ name: "《春秋公羊经传解诂》", src: "《春秋公羊经传解诂》.jpg", title: "近<span class='yan-char'>颜</span>字《多宝塔》字体", desc: "宋淳熙抚州公使库刻绍熙四年重修本《春秋公羊经传解诂》采用了一种方正平缓、布白均衡、近于颜字《多宝塔》的字体。" }] }, type: EventType.Contents },
                { event: { name: "庆元", year: 1195 }, type: EventType.Year },
                { event: { name: "宋宁宗赵扩", year: 1195, src: "赵扩.png" }, type: EventType.Empire },
                { event: { year: 1195, len: 1, contents: [{ name: "《中庸辑略》", src: "《中庸辑略》.jpg", title: "扁方<span class='ou-char'>欧</span>体", desc: "宋庆元元年至嘉定十七年刻本《中庸辑略》，体势扁肥而线条粗壮。" }] }, type: EventType.Contents },
                { event: { year: 1196, len: 1, contents: [{ name: "《欧阳文忠公集》", src: "《欧阳文忠公集》.png", title: "有北齐刻经意韵", desc: "南宋庆元二年周必大吉安刻本《欧阳文忠公集》的部分字体，有北齐刻经意韵，字体开张，长枪大戟，疏宕古隽，堪称宋代版刻楷书中的一朵奇葩。" }] }, type: EventType.Contents },
                { event: { name: "嘉泰", year: 1201 }, type: EventType.Year },
                { event: { name: "开禧", year: 1205 }, type: EventType.Year },
            ],
            span: [1164, 1205],
            desc: "南宋中期"
        },
        // 8
        {
            events: [
                { event: { year: 1207, len: 1, contents: [{ name: "《昆山杂咏》", src: "《昆山杂咏》.jpg", title: "近于行楷的字体", desc: "宋开禧三年昆山县斋刻本《昆山杂咏》以行楷笔意写稿，多有连笔及简写处。提按顿挫，笔意毕现。结字流美而富有新意，绝非俗手所书。长枪大戟，疏宕古隽，堪称宋代版刻楷书中的一朵奇葩。" }] }, type: EventType.Contents },
                { event: { name: "嘉定", year: 1208 }, type: EventType.Year },
                { event: { year: 1213, len: 1, contents: [{ name: "《周髀筭经》", src: "《周髀筭经》.png", title: "字体略扁<span class='ou-char'>欧</span>", desc: "原本鲜活的欧字，逐步向程式化方向发展。这类风格占据了现存南宋中期浙本书籍的大多数，如：宋嘉定十三年陆子遹溧阳学宫刻本《渭南文集》。:3的悬殊差别，除了“曰”等字夸张的“口”字形尚有些颜体特征外，已很难找到原本靠近欧、颜的影子。" }] }, type: EventType.Contents },
                { event: { year: 1220, len: 1, contents: [{ name: "《渭南文集》", src: "《渭南文集》.jpg", title: "程式化", desc: "字体略扁横竖的粗细变化加大，形成了横竖之比为1:3的悬殊差别，除了“曰”等字夸张的“口”字形尚有些颜体特征外，已很难找到原本靠近欧、颜的影子。" }] }, type: EventType.Contents },
                { event: { name: "宝庆", year: 1225 }, type: EventType.Year },
                { event: { name: "宋理宗赵昀", year: 1225, src: "赵昀.png" }, type: EventType.Empire },
                { event: { year: 1226, len: 1, contents: [{ name: "《东汉会要》", src: "《东汉会要》.png", title: "靠近<span class='ou-char'>欧</span>、<span class='liu-char'>柳</span>等唐楷字体", desc: "宋宝庆二年建宁郡斋刻本《东汉会要》刀法变圆，风棱减弱，靠近欧、柳等唐楷字体日本长泽规矩也云：“到了宋末，浙中因遭受战祸而衰退，建安的出版业则有较大的发展。宋末的建安刊本，文字加一层圆味，每行字数的排列变多，字形却变小；盖为了大量销售，必须力求降低价格。此后风气一直到元代，故宋末元初的建本难予鉴别。”" }] }, type: EventType.Contents },
                { event: { name: "绍定", year: 1128 }, type: EventType.Year },
                { event: { name: "端平", year: 1234 }, type: EventType.Year },
                { event: { name: "嘉熙", year: 1237 }, type: EventType.Year },
                { event: { name: "淳祐", year: 1241 }, type: EventType.Year },
                { event: { name: "宝祐", year: 1253 }, type: EventType.Year },
                { event: { name: "赵孟頫", year: 1254, end: 1322, src: "赵孟頫.png", desc: "“楷书四大家”；晚年作品最能代表赵氏书风，其中楷、行为最；在中国书法艺术史上有着不可忽视的重要作用和深远的影响力" }, type: EventType.Artist },
                { event: { name: "开庆", year: 1259 }, type: EventType.Year },
                { event: { name: "景定", year: 1260 }, type: EventType.Year },
                { event: { name: "咸淳", year: 1265 }, type: EventType.Year },
                { event: { name: "宋度宗赵禥", year: 1265, src: "赵禥.png" }, type: EventType.Empire },
                { event: { year: 1267, len: 1, contents: [{ name: "《邵子观物篇渔樵问对》", src: "《邵子观物篇渔樵问对》.png", title: "字体有圆意味", desc: "南宋晚期，福建地区版刻书法风格向两个方向发展。一是在中期建本大字的基础上，强化刀法，线条愈加方硬，转弯处却加上一层圆意。如宋咸淳福建漕治刻《邵子观物篇渔樵问对》" }] }, type: EventType.Contents },
                { event: { year: 1273, len: 1, contents: [{ name: "《无文印》", src: "《无文印》.jpg", title: "书棚体", desc: "南宋晚期，一直盛行于浙江地区的欧体字已经演变成较为僵化的字体，成为明朝中叶出现的宋体字之渊薮。其字体横平竖直、横细竖粗、左右对称、方硬呆板，是书棚体的主要特征。以横平竖直为例，书棚体横竖的斜度与曲度变小，更加趋于严格意义上的横平竖直。从实用角度看，书棚体排列整齐、布局疏朗，从而使版面清洁美观，宜于阅读。书棚体除了以欧体为主要特征外，也有参以柳体特征者。" }] }, type: EventType.Contents },
            ],
            span: [1207, 1279],
            desc: "南宋晚期"
        },
        // 9
        {
            events: [
            ],
            span: [1271, 1368],
            desc: "元朝"
        }
    ]

    return events;
}