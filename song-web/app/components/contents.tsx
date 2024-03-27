'use client'

import { useEffect, useRef, useState } from "react";
import { EventType, Timeline, TimelineEvent } from "./timeline";

export function Contents() {
  const [introClass, setIntroClass] = useState('intro intro--idle');

  const introEnds = () => {
    setIntroClass('intro intro--exit intro--idle');
    setTimeout(() => {
      document.body.classList.remove('visual-intro-active');
    }, 1500);
  };


  return (
    <>
      {/* <Intro introClass={introClass} introEnds={introEnds}/> */}
      <Story />
    </>
  )
}

interface IntroProps {
  introClass: string;
  introEnds: () => void;
}

function Intro({ introClass, introEnds }: IntroProps) {
  return (
    <section className={introClass} onClick={introEnds}>
      <div className="intro-area intro-area--top">
      </div>
      <div className="intro__text-container intro__text-container--left">
        <div className="intro__text">
          {/* <div className="intro__date intro__date--small-screen">557 - 1368</div> */}
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
      <div className="intro__text-container intro__text-container--right">
        <h2 className="text-style heading1">
          <span className="intro__text intro-text__title intro-text__title--first">
            楷书书体
          </span>
          <span className="intro__text intro-text__title intro-text__title--second">
            演变史
          </span>
        </h2>
        <h3 className="text-style heading3">
          <span className="intro__text intro-text__sub-title intro-text__sub-title--first">
            两宋时期
          </span>
          <span className="intro__text intro-text__sub-title intro-text__sub-title--second">
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

function Story() {
  const events: TimelineEvent[] = [
    { name: 'test1', year: 55, ty: EventType.Empire },
  ];

  return (
    <div className="story-container">
      <div className="buttons">
        <button className="previous">Previous</button>
        <button className="next">Next</button>
      </div>
      <div id="chart-wrapper">
        <div className="event-box-container">
          <div className="event-box">
            <p className="event-date">AD 000</p>
            <p className="event-title">测试文字</p>
            <p className="event-content">测试文字</p>
            <p className="event-content">
              测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字
            </p>
            <img className="event-img" src="/test.jpg" />
          </div>
        </div>
        <div className="chart-container">
          <Timeline events={events} span={[0, 100]} />
        </div>
      </div>
      <div className="legends">
        <p className="legends-title">数据选择</p>
        <ul className="legend-container">
          <li className="legend-item">
            <input type="checkbox" className="checkbox-empire" id="checkbox-empire" name="checkbox-empire" ></input>
            <label htmlFor="checkbox-empire">宋朝皇帝</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-year" id="checkbox-year" name="checkbox-year" ></input>
            <label htmlFor="checkbox-year">宋朝年号</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-artist" id="checkbox-artist" name="checkbox-artist" ></input>
            <label htmlFor="checkbox-artist">楷书书法家</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-ou" id="checkbox-ou" name="checkbox-ou" ></input>
            <label htmlFor="checkbox-ou">欧体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-yan" id="checkbox-yan" name="checkbox-yan" ></input>
            <label htmlFor="checkbox-yan">颜体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-liu" id="checkbox-liu" name="checkbox-liu" ></input>
            <label htmlFor="checkbox-liu">柳体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-zhao" id="checkbox-zhao" name="checkbox-zhao" ></input>
            <label htmlFor="checkbox-zhao">赵体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-others" id="checkbox-others" name="checkbox-others" ></input>
            <label htmlFor="checkbox-others">其他体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-overview" id="checkbox-overview" name="checkbox-overview" ></input>
            <label htmlFor="checkbox-overview">南宋早中后期楷体概述</label>
          </li>
        </ul>
      </div>
    </div>
  )
}