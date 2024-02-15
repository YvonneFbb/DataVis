export function Intro() {
  return (
    <section className="intro intro--idle">
      <div className="intro-area intro-area--top">
        <div className="intro-slice intro-slice--1"></div>
        <div className="intro-slice intro-slice--2"></div>
        <div className="intro-slice intro-slice--3"></div>
        <div className="intro-slice intro-slice--4"></div>
      </div>
      <div className="intro-text-container intro-text-container--left">
        <div className="intro-text">
          <div className="intro__date intro__date--small-screen">557 - 1368</div>
          <div className="intro-text__paragraph">
            <p className="text-style p">
              宋代刻本之美在于字体、板式。我们特地整理分析了宋刻板中楷书字体的发展时间史及江浙、四川、福建三地宋刻本中楷书字体的异同和字体倾向，以数据可视化的形式展示楷书及宋刻本书体演变史。
            </p>
          </div>
        </div>
        <div className="intro__date intro__date--large-screen">
          1368
        </div>
      </div>
      <div className="intro-area intro-area--left"></div>
      <div className="intro-area intro-area--bottom"></div>
      <div className="intro-text-container intro-text-container--right">
        <h2 className="text-style heading1">
          <span className="intro-text intro-text__title intro-text__title--first">
            楷书书体
          </span>
          <span className="intro-text intro-text__title intro-text__title--second">
            演变史
          </span>
        </h2>
        <h3 className="text-style heading3">
          <span className="intro-text intro-text__sub-title intro-text__sub-title--first">
            两宋时期
          </span>
          <span className="intro-text intro-text__sub-title intro-text__sub-title--second">
            宋版书
          </span>
        </h3>
        <div className="intro__date intro__date--large-screen">
          557
        </div>
      </div>
      <div className="intro-area intro-area--right"></div>
    </section>
  )
}