'use client'

import * as d3 from 'd3';
import { useEffect, useRef, useState } from "react";

import { IntroCanvas } from "./canvas";
import { EventType, LoadEvents, TimelineEvent, TimeLineEvents } from "./events";
import { Timeline } from "./timeline";

export function Contents() {
  return (
    <>
      <Intro />
      {/* <Story /> */}
    </>
  )
}

function Intro() {
  return (
    <div style={{ width: '100vw', height: '100vh', position: 'relative', backgroundColor: '#161616' }}>
      <IntroCanvas />
      <div className='intro'>
        <div className='intro-desc-container'>
          <div className='intro-desc-left'>
            <p className='intro-desc-text'>
              宋代刻本之美在于字体、板式。我们特地整理分析了宋刻板中楷书字体的发展时间史及江浙、四川、福建三地宋刻本中楷书字体的异同和字体倾向，以数据可视化的形式展示楷书及宋刻本书体演变史。
            </p>
          </div>
          <div className='intro-desc-divider'></div>
          <div className='intro-desc-right'>
            <p className='intro-desc-text'>
              两宋时期宋版书
            </p>
          </div>
        </div>
        <div className='intro-title-container'>
          <p className='intro-title-text'>
            演变史
          </p>
          <p className='intro-title-text'>
            楷书书体
          </p>
        </div>
      </div>
    </div>
  );
}

// function Intro() {
//   const introRef = useRef<HTMLDivElement>(null);

//   const handleOnClick = () => {
//     const intro = d3.select(introRef.current);
//     intro.transition().duration(800).style('opacity', 0)
//       .end().then(() => {
//         intro.style('display', 'none');
//       });
//   }

//   return (
//     <div ref={introRef} className='intro intro--idle' onClick={handleOnClick}>
//       <div className="intro-area intro-area--top">
//       </div>
//       <div className="intro__text-container intro__text-container--left">
//         <div className="intro__text">
//           <div className="intro-text__paragraph">
//             <p className="text-style p">
//               宋代刻本之美在于字体、板式。我们特地整理分析了宋刻板中楷书字体的发展时间史及江浙、四川、福建三地宋刻本中楷书字体的异同和字体倾向，以数据可视化的形式展示楷书及宋刻本书体演变史。
//             </p>
//           </div>
//         </div>
//         <div className="intro__date intro__date--large-screen">
//           1368
//         </div>
//       </div>
//       <div className="intro-area intro-area--left"></div>
//       <div className="intro-area intro-area--bottom"></div>
//       <div className="intro__text-container intro__text-container--right">
//         <h2 className="text-style heading1">
//           <span className="intro__text intro-text__title intro-text__title--first">
//             楷书书体
//           </span>
//           <span className="intro__text intro-text__title intro-text__title--second">
//             演变史
//           </span>
//         </h2>
//         <h3 className="text-style heading3">
//           <span className="intro__text intro-text__sub-title intro-text__sub-title--first">
//             两宋时期
//           </span>
//           <span className="intro__text intro-text__sub-title intro-text__sub-title--second">
//             宋版书
//           </span>
//         </h3>
//         <div className="intro__date intro__date--large-screen">
//           557
//         </div>
//       </div>
//       <div className="intro-area intro-area--right"></div>
//     </div>
//   )
// }

function Story() {
  const events = LoadEvents();
  const maxPage = events.length;
  const [currentPage, setCurrentPage] = useState(0)

  const switchRef = useRef<HTMLDivElement>(null);

  const do_switch = (direction: number) => {
    if (direction == 0) {
      const swp = d3.select(switchRef.current).select('.previous');
      const swn = d3.select(switchRef.current).select('.next');

      swp.style('border-left-width', '200vw');
      setTimeout(() => {
        swn.style('border-right-width', '200vw');
      }, 400)
      setTimeout(() => {
        swp.style('border-left-width', '0');
      }, 800)
      setTimeout(() => {
        swn.style('border-right-width', '0');
      }, 1200)
    } else {
      const swp = d3.select(switchRef.current).select('.previous');
      const swn = d3.select(switchRef.current).select('.next');

      swn.style('border-right-width', '200vw');
      setTimeout(() => {
        swp.style('border-left-width', '200vw');
      }, 400)
      setTimeout(() => {
        swn.style('border-right-width', '0');
      }, 800)
      setTimeout(() => {
        swp.style('border-left-width', '0');
      }, 1200)
    }
  };

  return (
    <div className="story-container">
      <div ref={switchRef} className='switch'>
        <div className='previous'></div>
        <div className='next'></div>
      </div>
      <div className="buttons">
        {currentPage > 0 ? (
          <button className="previous" onClick={() => { do_switch(0); setTimeout(() => { setCurrentPage(currentPage - 1); }, 800) }}>Previous</button>
        ) : (
          <button disabled>Previous</button>
        )}
        {currentPage < maxPage - 1 ? (
          <button className="next" onClick={() => { do_switch(1); setTimeout(() => { setCurrentPage(currentPage + 1); }, 800) }}>Next</button>
        ) : (
          <button disabled>Next</button>
        )}
      </div>
      <div id="chart-wrapper">
        <div className="chart-container">
          <div>
            <div className="dash-line-1">
              <span className="dashed-line-text">宋刻本</span>
            </div>
            <div className="dash-line-2"></div>
            <div className="dash-line-3">
              <span className="dashed-line-text">概述</span>
            </div>
            <div className="dash-line-4">
              <span className="dashed-line-text">人物</span>
            </div>
          </div>
          <Timeline events={events[currentPage].events} span={events[currentPage].span} current={currentPage} />
        </div>
      </div>
    </div>
  )
}