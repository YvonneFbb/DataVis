'use client'

import * as d3 from 'd3';
import { useEffect, useRef, useState, createContext, useContext, MutableRefObject } from "react";

import { IntroCanvas } from "./canvas";
import { EventType, LoadEvents, TimelineEvent, TimeLineEvents } from "./events";
import { Timeline } from "./timeline";

interface OverallStatus {
  /* Canvas Status Usage */
  isLoaded: boolean;
  isFinalSelected: boolean;
  rotationSpeed: number;
  initCubeSize: number;
  cubeScale: number;
  selectedID: number;
}

export const OverallContext = createContext<MutableRefObject<OverallStatus>>({} as MutableRefObject<OverallStatus>);

export function Contents() {
  return (
    <>
      <Intro />
      {/* <Story /> */}
    </>
  )
}

function Intro() {

  const clickEntryButton = (id: number) => {
  }

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
        <div className='intro-button-container'>
          <button className='vertical-button vertical-button-first' onClick={() => { clickEntryButton(1) }}>开始探索时间轴</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(2) }}>南北朝</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(3) }}>隋朝</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(4) }}>唐朝</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(5) }}>宋朝</button>
          <button className='vertical-button' onClick={() => { clickEntryButton(6) }}>元朝</button>
        </div>
      </div>
    </div>
  );
}

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