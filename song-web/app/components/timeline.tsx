import * as d3 from 'd3';
import React, { useRef, useEffect, useState } from 'react';
import { ArtistEvent, ContentsEvent, EmpireEvent, EventType, TimelineEvent, YearEvent } from "./events";
import Image from 'next/image';

export type TimelineProps = {
  events: TimelineEvent[];
  span: [number, number];
  desc: string;
  current: number;
};

export function Timeline({ events, span, desc, current }: TimelineProps) {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [windowHeight, setWindowHeight] = useState(window.innerHeight);
  const [sliderOffset, setSliderOffset] = useState(0);
  const [shownStatus, setShownStatus] = useState(Array(9).fill(false));
  const [keyContent, setKeyContent] = useState(null);

  const svgRef = useRef<SVGSVGElement>(null);
  const focusRef = useRef<SVGGElement>(null);
  const viewRef = useRef<SVGGElement>(null);
  const eventBoxRef = useRef<HTMLDivElement>(null);
  const eventBgRef = useRef<HTMLDivElement>(null);
  const eventZoomRef = useRef<HTMLImageElement>(null);
  const artistBoxRef = useRef<HTMLImageElement>(null);
  const overviewBoxRef = useRef<HTMLImageElement>(null);

  const windowWidthRef = useRef(windowWidth);
  const sliderWidthRef = useRef(window.innerWidth / 6);
  const xScaleRef = useRef(d3.scaleLinear());

  const viewHeight = 85;
  const sliderHeight = viewHeight - 55;
  const focusHeight = windowHeight - viewHeight;
  const attachDistance = 30;
  const srcPath = "/timelines/" + desc + "/";

  const updateShownEvents = (ty: number) => {
    shownStatus[ty] = !shownStatus[ty];
    setShownStatus(shownStatus);

    const events = d3.select(viewRef.current).select('.timeline-events').selectAll('g.event')
      .filter((d) => (d as TimelineEvent).type === ty);

    if (shownStatus[ty]) {
      switch (ty) {
        case 3: eventBoxRef.current?.classList.add('ou-enable'); break;
        case 4: eventBoxRef.current?.classList.add('yan-enable'); break;
        case 5: eventBoxRef.current?.classList.add('liu-enable'); break;
        case 6: eventBoxRef.current?.classList.add('zhao-enable'); break;
        case 7: break;
        default: {
          events
            .attr('display', null)
            .transition()
            .duration(300)
            .style('opacity', 1);
        } break;
      }
    } else {
      switch (ty) {
        case 3: eventBoxRef.current?.classList.remove('ou-enable'); break;
        case 4: eventBoxRef.current?.classList.remove('yan-enable'); break;
        case 5: eventBoxRef.current?.classList.remove('liu-enable'); break;
        case 6: eventBoxRef.current?.classList.remove('zhao-enable'); break;
        case 7: break;
        default: {
          events
            .transition()
            .duration(300)
            .style('opacity', 0)
            .end()
            .then(() => {
              events.attr('display', 'none');
            });
        } break;
      }

    }
  }

  const updateSliderPosition = (newX: number, animation: boolean) => {
    const slider = d3.select(viewRef.current).select('.slider');
    const leftyear = d3.select(viewRef.current).select('.texts-decor').select('.leftyear');
    const rightyear = d3.select(viewRef.current).select('.texts-decor').select('.rightyear');

    slider.attr('x', newX);
    if (animation) {
      slider.transition().duration(300).attr('transform', `translate(${newX}, 0)`);
    } else {
      slider.attr('transform', `translate(${newX}, 0)`);
    }
    leftyear.text(xScaleRef.current.invert(newX).toFixed(0))
    rightyear.text(xScaleRef.current.invert(newX + sliderWidthRef.current).toFixed(0))
  }

  const updateZoomImg = () => {
    const eventBox = d3.select(eventBoxRef.current);
    if (eventBox.style('opacity') != '1') {
      return;
    }

    const eventZoomImg = d3.select(eventZoomRef.current);

    if (eventZoomImg.style('visibility') == 'visible') {
      eventZoomImg.style('opacity', 0);
      setTimeout(() => { eventZoomImg.style('visibility', 'hidden'); }, 400);
    } else {
      eventZoomImg.style('visibility', 'visible');
      eventZoomImg.style('opacity', 1);
    }
  }

  // Update window width and height
  useEffect(() => {
    const handleResize = () => {
      setWindowWidth(window.innerWidth);
      setWindowHeight(window.innerHeight);

      windowWidthRef.current = window.innerWidth;
      sliderWidthRef.current = window.innerWidth / 6;
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Initialize the chart
  useEffect(() => {
    const focus = d3.select(focusRef.current);
    const view = d3.select(viewRef.current);

    focus.selectAll('*').remove();
    view.selectAll('*').remove();

    setKeyContent(null);

    // Initialize the focus part
    focus.append('rect')
      .attr('class', 'focus-bg')
      .attr('width', windowWidth)
      .attr('height', focusHeight)
      .attr('visibility', 'hidden');

    // Add divider
    view.append('line')
      .attr('class', 'divider-line')
      .attr('x1', 0)
      .attr('y1', focusHeight)
      .attr('x2', windowWidth)
      .attr('y2', focusHeight)
      .attr('stroke', '#A5A5A5')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '4, 2'); // 设置虚线样式，4px线段，2px间隔

    // Initialize the view part
    view.append('rect')
      .attr('class', 'view-bg')
      .attr('width', windowWidth)
      .attr('height', viewHeight)
      .attr('transform', `translate(0,${focusHeight})`)
      .attr('fill', '#fff');

    const viewG = view.append('g')
      .attr('class', 'view-g')
      .attr('transform', `translate(0,${focusHeight})`);

    xScaleRef.current = d3.scaleLinear()
      .domain(span)
      .range([0 + sliderWidthRef.current / 2, windowWidth - sliderWidthRef.current / 2]);

    const eventsGroup = viewG.append('g')
      .attr('class', 'timeline-events')
      .attr('pointer-events', 'auto');

    eventsGroup.selectAll('g.event')
      .data(events)
      .join('g')
      .attr('class', 'event')
      .attr('transform', d => `translate(${xScaleRef.current(d.event.year)}, ${sliderHeight / 2})`)
      .attr('opacity', '0')
      .attr('display', 'none')
      .each(function (d) {
        const eventGroup = d3.select(this);

        switch (d.type) {
          case EventType.Empire: {
            const event = d.event as EmpireEvent;

            eventGroup.append('circle')
              .attr('r', 6)
              .attr('fill', 'none')
              .attr('stroke', '#A2483D')
              .attr('stroke-width', 1.5)
              .attr('style', 'cursor: pointer;');

            eventGroup.append('image')
              .attr('style', 'cursor: pointer;')
              .attr("href", srcPath + event.src)
              .attr("width", 90)
              .attr("height", 90)
              .attr("x", -45)
              .attr("y", -160);

            eventGroup.append('text')
              .attr('x', 27)
              .attr('y', -145)
              .text(event.name)
              .attr('style', 'font-size: 10px; font-family: \'MyCN_FZ_B\'; writing-mode: vertical-rl;')
              .attr('fill', '#fff');

            if (shownStatus[0]) {
              eventGroup.attr('opacity', '1')
                .attr('display', null);
            }
          }; break;
          case EventType.Year: {
            const event = d.event as YearEvent;

            eventGroup.append('circle')
              .attr('r', 3.5)
              .attr('fill', '#3da5f1')
              .attr('style', 'cursor: pointer;');

            eventGroup.append('text')
              .attr('class', 'flip-fill-text')
              .attr('x', 0)
              .attr('y', -62)
              .text(event.name)
              .attr('style', 'font-size: 10px; font-family: \'MyCN_FZ\'; writing-mode: vertical-rl; cursor: pointer;');

            if (shownStatus[1]) {
              eventGroup.attr('opacity', '1')
                .attr('display', null);
            }
          }; break;
          case EventType.Artist: {
            const event = d.event as ArtistEvent;

            eventGroup.append('circle')
              .attr('r', 6)
              .attr('fill', '#374C12')
              .attr('stroke-width', 1.5)
              .attr('style', 'cursor: pointer;');

            eventGroup.append('image')
              .attr('style', 'cursor: pointer;')
              .attr("href", srcPath + event.src)
              .attr("width", 90)
              .attr("height", 90)
              .attr("x", -45)
              .attr("y", -200);

            eventGroup.append('text')
              .attr('class', 'flip-fill-text')
              .attr('x', -18)
              .attr('y', -90)
              .text(event.name)
              .attr('style', 'font-size: 12px; font-family: \'MyCN_F_12B\';');

            eventGroup.append('text')
              .attr('class', 'flip-fill-text')
              .attr('x', -10)
              .attr('y', -75)
              .text(event.year)
              .attr('style', 'font-size: 12px; font-family: \'MyEN_DIN_Bold\';');

            if (shownStatus[2]) {
              eventGroup.attr('opacity', '1')
                .attr('display', null);
            }
          }; break;
          // case EventType.Ou: {

          // }; break;
          // case EventType.Yan: {

          // }; break;
          // case EventType.Liu: {

          // }; break;
          // case EventType.Zhao: {

          // }; break;
          // case EventType.Others: {

          // }; break;
          // case EventType.Overview: {

          // }; break;
          case EventType.Contents: {
            eventGroup.append('circle')
              .attr('r', 6)
              .attr('fill', '#A2483D')
              .attr('style', 'cursor: pointer;');

            eventGroup.attr('opacity', '1')
              .attr('display', null);
          }
        }
      })
      .on('click', (event, d) => {
        const posX = xScaleRef.current(d.event.year) - sliderWidthRef.current / 2;
        updateSliderPosition(posX, true);

        setKeyContent(null);
        if (d.type == EventType.Contents || d.type == EventType.Artist) {
          setTimeout(() => { setKeyContent(d as any) }, 200);
        }
      });

    const slider = viewG.append('g')
      .attr('class', 'slider')
      .attr('transform', `translate(0, 0)`);

    slider.append('rect')
      .attr('class', 'box')
      .attr('transform', `translate(0, 0)`)
      .attr('width', `${sliderWidthRef.current}`)
      .attr('height', `${sliderHeight}`)
      .attr('style', 'cursor: move;');

    // Add lines
    const linesGroup = slider.append('g')
      .attr('class', 'lines-decor')
      .attr('transform', `translate(${sliderWidthRef.current / 2}, ${sliderHeight / 2})`);

    [-10, -6, -2, 2, 6, 10].forEach(offset => {
      linesGroup.append('line')
        .attr('x1', offset)
        .attr('y1', -10)
        .attr('x2', offset)
        .attr('y2', 10)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);
    });

    // Add text
    const textsGroup = slider.append('g')
      .attr('class', 'texts-decor')
      .attr('width', sliderWidthRef.current);

    const leftyear = textsGroup.append('text')
      .attr('class', 'leftyear')
      .attr('transform', `translate(10, ${sliderHeight / 2 + 5})`)
      .text(xScaleRef.current.invert(0).toFixed(0))
      .attr('style', 'font-size: 126x; font-family: \'MyEN_DIN_Bold\';')
      .attr('fill', '#fff');

    const rightyear = textsGroup.append('text')
      .attr('class', 'rightyear')
      .attr('transform', `translate(${sliderWidthRef.current - 40}, ${sliderHeight / 2 + 5})`)
      .text(xScaleRef.current.invert(sliderWidthRef.current).toFixed(0))
      .attr('style', 'font-size: 126x; font-family: \'MyEN_DIN_Bold\';')
      .attr('fill', '#fff');

    let dragOffset = 0;
    const drag = d3.drag()
      .on('start', function (event, d) {
        const currentX = d3.select(this).attr('x');
        dragOffset = event.x - (currentX as unknown as number);
        d3.select(this).classed('active', true);

        setKeyContent(null);
      })
      .on('drag', function (event, d) {
        let newX = event.x - dragOffset;
        newX = Math.max(0, Math.min(windowWidthRef.current - sliderWidthRef.current, newX));

        // move slider
        setSliderOffset(newX);
        updateSliderPosition(newX, false);
      })
      .on('end', function (event, d) {
        d3.select(this).classed('active', false);

        const currentX = (+ d3.select(this).attr('x') as unknown as number) + sliderWidthRef.current / 2;
        let closestDistance = Infinity;
        let closestEventX = null;
        let contents = null;

        d3.select(viewRef.current).selectAll('g.event')
          .filter(function (d) {
            return d3.select(this).attr('display') !== 'none' && ((d as TimelineEvent).type == EventType.Contents || (d as TimelineEvent).type == EventType.Artist);
          })
          .each(function (d) {
            const eventX = xScaleRef.current((d as TimelineEvent).event.year);
            const distance = Math.abs(eventX - currentX);
            if (distance < attachDistance && distance < closestDistance) {
              closestDistance = distance;
              closestEventX = eventX - sliderWidthRef.current / 2;
              contents = d;
            }
          });

        if (closestEventX !== null) {
          // move slider
          updateSliderPosition(closestEventX, true);
          setKeyContent(contents);
        }
      });

    slider.call(drag as any);
  }, [current])

  // Updates when resize
  useEffect(() => {
    const viewHeight = 85;
    const focusHeight = windowHeight - viewHeight;

    const focus = d3.select(focusRef.current);
    const view = d3.select(viewRef.current);

    focus.select('.focus-bg')
      .attr('width', windowWidth)
      .attr('height', focusHeight);

    view.select('.view-bg')
      .attr('width', windowWidth)
      .attr('height', viewHeight)
      .attr('transform', `translate(0,${focusHeight})`);

    view.select('.view-g')
      .attr('transform', `translate(0,${focusHeight})`);

    xScaleRef.current = d3.scaleLinear()
      .domain(span)
      .range([0 + sliderWidthRef.current / 2, windowWidth - sliderWidthRef.current / 2]);

    view.select('.timeline-events').selectAll('g.event')
      .each(function (data, index) {
        const event = d3.select(this);
        const newX = xScaleRef.current((data as TimelineEvent).event.year);

        event.attr('transform', `translate(${xScaleRef.current((data as TimelineEvent).event.year)}, ${sliderHeight / 2})`);
      });

    view.select('.slider .box')
      .attr('width', `${sliderWidthRef.current}`)
      .attr('height', `${sliderHeight}`);

    view.select('.lines-decor')
      .attr('transform', `translate(${sliderWidthRef.current / 2}, ${sliderHeight / 2})`);

    view.select('.rightyear')
      .attr('transform', `translate(${sliderWidthRef.current - 40}, ${sliderHeight / 2 + 5})`)

    updateSliderPosition(0, true);

  }, [windowWidth, windowHeight])

  const [selectMultiCont, setMultiCont] = useState(0);
  useEffect(() => {
    if (keyContent != null) {
      const event = keyContent as TimelineEvent;
      if (event.type == EventType.Contents) {
        const contents = event.event as ContentsEvent;
        const eventBox = d3.select(eventBoxRef.current);
        const eventBgImg = d3.select(eventBgRef.current);
        const eventZoomImg = d3.select(eventZoomRef.current);
        const eventX = xScaleRef.current(contents.year);

        var selectContent = contents.contents[selectMultiCont];

        eventBox.select('.event-date').text('AD ' + contents.year);
        eventBox.select('.event-title').text(selectContent.name);
        eventBox.select('.event-subtitle').html(selectContent.title);
        eventBox.select('.event-content').text(selectContent.desc);
        eventBox.select('.event-img').attr('src', srcPath + selectContent.src);

        if (contents.len <= 1) {
          eventBox.select('.event-button.up').style('visibility', 'hidden');
          eventBox.select('.event-button.down').style('visibility', 'hidden');
          eventBox.select('.event-buttons .page-number').style('visibility', 'hidden');
        } else {
          eventBox.select('.event-buttons .page-number').style('visibility', 'visible');
          eventBox.select('.event-button.up').style('visibility', 'visible');
          eventBox.select('.event-button.down').style('visibility', 'visible');
          if (selectMultiCont == 0) {
            eventBox.select('.event-button.up').style('visibility', 'hidden');
          } else if (selectMultiCont == contents.len - 1) {
            eventBox.select('.event-button.down').style('visibility', 'hidden');
          }
        }
        eventBgImg.select('.event-bg-img').attr('src', srcPath + selectContent.src);
        eventZoomImg.select('.event-zoom-img').attr('src', srcPath + selectContent.src);

        if (eventX > 0.5 * windowWidth) {
          eventBox.style('left', (eventX - 0.42 * windowWidth) + 'px');
        } else {
          eventBox.style('left', eventX + 'px');
        }
        eventBox.select('.event-vertical-line').style('left', eventX + 'px');

        eventBox.style('opacity', 1);
        eventBgImg.style('opacity', 1);
        document.getElementById('body')?.classList.add('content', 'white');
      } else if (event.type == EventType.Artist) {
        const artist = event.event as ArtistEvent;
        const artistBox = d3.select(artistBoxRef.current);
        const eventX = xScaleRef.current(artist.year);

        artistBox.select('.artist-title').text(artist.name);
        artistBox.select('.artist-date').text('AD ' + artist.year + '-' + artist.end);
        artistBox.select('.artist-content').text(artist.desc);

        artistBox.style('left', eventX - 50 - (Number(artistBox.style('width').replace('px', ''))) + 'px');
        artistBox.style('opacity', 1);
      }
    } else {
      const eventBox = d3.select(eventBoxRef.current);
      const eventBgImg = d3.select(eventBgRef.current);
      const artistBox = d3.select(artistBoxRef.current);

      setMultiCont(0);

      eventBox.style('opacity', 0);
      eventBgImg.style('opacity', 0);
      artistBox.style('opacity', 0);

      document.getElementById('body')?.classList.remove('content', 'white');
    }
  }, [keyContent, selectMultiCont])

  return (
    <>
      <div ref={eventZoomRef} className="event-zoom" style={{ 'opacity': 0, 'visibility': 'hidden' }} onClick={updateZoomImg}>
        <img className="event-zoom-img" src="/test.jpg" alt="zooming" />
      </div>
      {/* <div ref={overviewBoxRef} className="overview-box" style={{ 'opacity': 1 }}>
      </div> */}
      <div ref={artistBoxRef} className="artist-box" style={{ 'opacity': 0 }}>
        <div className='artist-title'>测试文字</div>
        <div className='artist-date'>AD 000</div>
        <div className='artist-content'>测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字</div>
      </div>
      <div ref={eventBoxRef} className="event-box" style={{ 'opacity': 0 }}>
        <div className='event-vertical-line'></div>
        <div className="event-buttons">
          <button className="event-button up" onClick={() => setMultiCont(selectMultiCont <= 0 ? 0 : selectMultiCont - 1)}>↑</button>
          <span className="page-number">{selectMultiCont + 1}</span>
          <button className="event-button down" onClick={() => setMultiCont(selectMultiCont + 1)}>↓</button>
        </div>
        <p className="event-date">AD 000</p>
        <p className="event-title">测试文字</p>
        <p className="event-subtitle">测试文字</p>
        <p className="event-content">
          测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字测试文字
        </p>
        <img className="event-img" src="/test.jpg" alt="" onClick={updateZoomImg} />
      </div>
      <div ref={eventBgRef} className="event-bg" style={{ 'opacity': 0 }}>
        <img className="event-bg-img" src="/test.jpg" alt="" />
      </div>
      <svg ref={svgRef} width={windowWidth} height={windowHeight}>
        <g ref={focusRef} className='focus' />
        <g ref={viewRef} className='view' />
      </svg>
      <div className="legends">
        <p className="legends-title">数据选择：</p>
        <ul className="legend-container">
          <li className="legend-item">
            <input type="checkbox" className="checkbox-empire" id="checkbox-empire" /*checked={shownStatus[0]}*/ onClick={() => { updateShownEvents(0) }}></input>
            <label htmlFor="checkbox-empire">宋朝皇帝</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-year" id="checkbox-year" /*checked={shownStatus[1]}*/ onClick={() => { updateShownEvents(1) }}></input>
            <label htmlFor="checkbox-year">宋朝年号</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-artist" id="checkbox-artist" /*checked={shownStatus[2]}*/ onClick={() => { updateShownEvents(2) }}></input>
            <label htmlFor="checkbox-artist">楷书书法家</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-ou" id="checkbox-ou" /*checked={shownStatus[3]}*/ onClick={() => { updateShownEvents(3) }}></input>
            <label htmlFor="checkbox-ou">欧体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-yan" id="checkbox-yan" /*checked={shownStatus[4]}*/ onClick={() => { updateShownEvents(4) }}></input>
            <label htmlFor="checkbox-yan">颜体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-liu" id="checkbox-liu" /*checked={shownStatus[5]}*/ onClick={() => { updateShownEvents(5) }}></input>
            <label htmlFor="checkbox-liu">柳体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-zhao" id="checkbox-zhao" /*checked={shownStatus[6]}*/ onClick={() => { updateShownEvents(6) }}></input>
            <label htmlFor="checkbox-zhao">赵体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-others" id="checkbox-others" /*checked={shownStatus[7]}*/ onClick={() => { updateShownEvents(7) }}></input>
            <label htmlFor="checkbox-others">其他体</label>
          </li>
          <li className="legend-item">
            <input type="checkbox" className="checkbox-overview" id="checkbox-overview" /*checked={shownStatus[8]}*/ onClick={() => { updateShownEvents(8) }}></input>
            <label htmlFor="checkbox-overview">楷体概述</label>
          </li>
        </ul>
      </div>
    </>
  );
}
