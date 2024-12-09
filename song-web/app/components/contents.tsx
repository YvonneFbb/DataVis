'use client'

import * as d3 from 'd3';
import { useRef, useState, createContext, useContext, MutableRefObject, forwardRef, useEffect } from "react";

import { charDescGroups, IntroCanvas } from "./canvas";
import { LoadEvents, TimelineEvent, } from "./events";
import { Timeline } from "./timeline";

interface OverallStatus {
  /* Intro Status */
  isIntroEnd: boolean;
  introRef: React.RefObject<HTMLDivElement>;

  /* Canvas Status Usage */
  isCanvasLoaded: boolean;
  rotationSpeed: number;
  initCubeSize: number;
  cubeScale: number;
  selectedID: number;
  selectedGID: number;
  selectLevel: number;

  /* Store Status */
  storeRef: React.RefObject<HTMLDivElement>;
  currentPage: number;

}

export const OverallContext = createContext<MutableRefObject<OverallStatus>>({} as MutableRefObject<OverallStatus>);

export function Contents() {
  const overallStatus = useRef<OverallStatus>({
    /* Intro Status */
    isIntroEnd: false,
    introRef: useRef<HTMLDivElement>(null),
    /* Canvas Status Usage */
    isCanvasLoaded: false,
    // rotationSpeed: 0.1,
    rotationSpeed: 0.0,
    initCubeSize: 2.5,
    cubeScale: 1,
    selectedID: -1,
    selectedGID: -1,
    selectLevel: 0,

    /* Store Status */
    storeRef: useRef<HTMLDivElement>(null),
    currentPage: 0,
  });


  const [isIntroEnd, setIsIntroEnd] = useState(false);
  const introShallEnd = () => {
    console.log('setting');
    setIsIntroEnd(true);
  }

  useEffect(() => {
    if (isIntroEnd) {
    } else {
      document.getElementById('status-bar')!.innerText = "选择「永」字笔画探索时间轴";
    }
  }, [isIntroEnd]);

  return (
    <OverallContext.Provider value={overallStatus}>
      {!isIntroEnd && <Intro shallEnd={introShallEnd} ref={overallStatus.current.introRef} />}
      {isIntroEnd && <Story shallEnd={introShallEnd} ref={overallStatus.current.storeRef} />}
      {/* <Story shallEnd={introShallEnd} ref={overallStatus.current.storeRef} /> */}
    </OverallContext.Provider>
  )
}

interface ContProps {
  shallEnd: () => void;
}

const Intro = forwardRef<HTMLDivElement, ContProps>(({ shallEnd }, ref) => {
  const overallStatus = useContext(OverallContext);

  const clickEntryButton = (id: number) => {
    if (id == 0) {
      // Click the final view button
      const introRef = overallStatus.current.introRef.current!;
      const charDesc = charDescGroups[(overallStatus.current.selectedGID)];

      overallStatus.current.currentPage = charDesc.page;

      introRef.style.opacity = '0';
      setTimeout(() => {
        shallEnd();
        setTimeout(() => {
          const storeRef = overallStatus.current.storeRef.current!;
          storeRef.style.opacity = '1';
        }, 100);
      }, 500);
    } else if (id == 1) {
      overallStatus.current.selectLevel = 2;
      overallStatus.current.selectedGID = Math.floor(Math.random() * 6);

      // 设置介绍文字
      const introRef = overallStatus.current.introRef.current!;
      const charDesc = charDescGroups[(overallStatus.current.selectedGID)];

      (introRef.querySelector('.intro-chardesc-title') as HTMLElement).innerText = charDesc.title;
      (introRef.querySelector('.intro-chardesc-subtitle') as HTMLElement).innerText = 'AD ' + charDesc.year;
      (introRef.querySelector('.intro-chardesc-caption') as HTMLElement).innerText = charDesc.caption;
      (introRef.querySelector('.intro-chardesc-description') as HTMLElement).innerText = charDesc.desc;
      setTimeout(() => {
        const buttonBox = introRef.querySelector('.intro-button-container') as HTMLElement;
        buttonBox.classList.remove('show');
      }, 1000 * 1);
      setTimeout(() => {
        const charBox = introRef.querySelector('.intro-chardesc-container') as HTMLElement;
        charBox.classList.add('show');
      }, 1000 * 2.5);
    } else {
      switch (id) {
        case 2: overallStatus.current.currentPage = 0; break;
        case 3: overallStatus.current.currentPage = 1; break;
        case 4: overallStatus.current.currentPage = 2; break;
        case 5: overallStatus.current.currentPage = 3; break;
        case 6: overallStatus.current.currentPage = 9; break;
      }

      const introRef = overallStatus.current.introRef.current!;
      introRef.style.opacity = '0';
      setTimeout(() => {
        shallEnd();
        setTimeout(() => {
          const storeRef = overallStatus.current.storeRef.current!;
          storeRef.style.opacity = '1';
        }, 100);
      }, 500);
    }
  }

  return (
    <div ref={ref} style={{ width: '100vw', height: '100vh', position: 'relative', backgroundColor: '#161616', transition: 'opacity 0.5s ease-in-out' }}>
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
        <div className="intro-chardesc-container">
          <h1 className="intro-chardesc-title">测试文字</h1>
          <h3 className="intro-chardesc-subtitle">AD 000</h3>
          <p className="intro-chardesc-caption">测试文字</p>
          <p className="intro-chardesc-description">
            测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字
          </p>
          <button className="intro-chardesc-button" onClick={() => { clickEntryButton(0) }}>跳转时间轴</button>
        </div>
        <div className="intro-button-container show">
          <button className='vertical-button vertical-button-first' onClick={() => { clickEntryButton(1) }}>开始探索时间轴</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(2) }}>南北朝</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(3) }}>隋朝</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(4) }}>唐朝</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(5) }}>宋朝</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(6) }}>元朝</button>
        </div>
        <div className='intro-corner-date'>
          <div className="corner-date-box left-corner">
            <span className="text">557</span>
          </div>
          <div className="corner-date-box right-corner">
            <span className="text">1368</span>
          </div>
        </div>
      </div>
    </div >
  );
});

Intro.displayName = "Intro";

const Story = forwardRef<HTMLDivElement, ContProps>(({ shallEnd }, ref) => {
  const overallStatus = useContext(OverallContext);

  const events = LoadEvents();
  const maxPage = events.length;
  const [currentPage, setCurrentPage] = useState(overallStatus.current.currentPage);

  const switchRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.getElementById('status-bar')!.innerText = events[currentPage].desc;
  }, [currentPage]);

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
    <div ref={ref} style={{ opacity: '1', transition: 'opacity 0.5s ease-in-out' }}>
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
            <Timeline events={events[currentPage].events} span={events[currentPage].span} desc={events[currentPage].desc} current={currentPage} />
          </div>
        </div>
      </div>
    </div>
  )
});

Story.displayName = "Story";