import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';
import { Selection } from 'd3';

export type TimelineEvent = {
  date: string;
  name: string;
};

type TimelineProps = {
  events: TimelineEvent[];
};

export function Timeline({ events }: TimelineProps) {
  const [sliderPosition, setSliderPosition] = useState(0);
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [windowHeight, setWindowHeight] = useState(window.innerHeight);
  const ref = useRef<SVGSVGElement>(null);

  // Update window width and height
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Initialize the chart
  useEffect(() => {
    const svg = d3.select(ref.current);

    // Clear the SVG element
    svg.selectAll('*').remove();

    const focus = svg.append('g').attr('class', 'focus');
    const view = svg.append('g').attr('class', 'view');

    const viewHeight = 85;
    const focusHeight = windowHeight - viewHeight;

    // Initialize the focus part
    focus.append('rect')
      .attr('class', 'focus-bg')
      .attr('width', windowWidth)
      .attr('height', focusHeight)
      .attr('fill', 'grey');
    // .attr('visibility', 'hidden');

    // Initialize the view part
    view.append('rect')
      .attr('class', 'view-bg')
      .attr('width', windowWidth)
      .attr('height', viewHeight)
      .attr('transform', `translate(0,${focusHeight})`)
      .attr('visibility', 'hidden');

    const viewG = view.append('g')
      .attr('class', 'view-g')
      .attr('transform', `translate(0,${focusHeight})`);

    const slider = viewG.append('g')
      .attr('class', 'slider');

    const sliderWidth = windowWidth / 6;
    const sliderHeight = viewHeight - 55;
    slider.append('rect')
      .attr('class', 'extent')
      .attr('x', 0)
      .attr('width', `${sliderWidth}`)
      .attr('height', `${sliderHeight}`)
      .attr('style', 'cursor: move;');

    // Add lines
    const linesGroup = slider.append('g')
      .attr('class', 'lines-decor')
      .attr('transform', `translate(${sliderPosition + sliderWidth / 2}, ${sliderHeight / 2})`);

    [-10, -6, -2, 2, 6, 10].forEach(offset => {
      linesGroup.append('line')
        .attr('x1', offset)
        .attr('y1', -10)
        .attr('x2', offset)
        .attr('y2', 10)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1);
    });

    // TODO: Add text
    // const textsGroup = slider.append('g')
    //   .attr('class', 'texts-decor');

    // Add drag behavior
    const drag = d3.drag()
      .on('start', function (event, d) {
        d3.select(this).classed('active', true);
      })
      .on('drag', function (event, d) {
        let x = event.x - sliderWidth / 2; // 减去宽度的一半以使鼠标位于滑块中心
        x = Math.max(0, Math.min(windowWidth - sliderWidth, x)); // 确保滑块不会移出SVG容器
        d3.select(this)
          .attr('transform', `translate(${x},0)`); // 沿着x轴移动滑块
      })
      .on('end', function (event, d) {
        d3.select(this).classed('active', false);
      });

    slider.call(drag as any);
  }, [events, windowWidth, windowHeight])

  return (
    <svg ref={ref} className='fbb-chart' width={windowWidth} height={windowHeight} />
  );

  // useEffect(() => {
  //   const svg = d3.select(ref.current);
  //   const margin = { top: 20, right: 30, bottom: 20, left: 30 };
  //   const width = windowWidth - margin.left - margin.right;
  //   const height = 100 - margin.top - margin.bottom;

  //   // 清除以前的 SVG 元素
  //   svg.selectAll('*').remove();

  //   // 创建时间比例尺
  //   const xScale = d3.scaleTime()
  //     .domain(d3.extent(events, (d) => new Date(d.date)) as [Date, Date])
  //     .range([0, width]);

  //   // 绘制滑块

  //   const slider = svg.append('g')
  //     .attr('class', 'slider')
  //     .attr('transform', `translate(${margin.left + sliderPosition},${margin.top})`);

  //   slider.append('rect')
  //     .attr('class', 'slider-rect')
  //     .attr('x', -5)
  //     .attr('width', 100)
  //     .attr('height', height)
  //     .attr('fill', 'red');

  //   // 在滑块两侧添加文本
  //   slider.append('text')
  //     .attr('class', 'slider-text left')
  //     .attr('x', -10)
  //     .attr('y', height / 2)
  //     .attr('dy', '.35em')
  //     .attr('text-anchor', 'end')
  //     .text(xScale.invert(sliderPosition).getFullYear());

  //   slider.append('text')
  //     .attr('class', 'slider-text right')
  //     .attr('x', 10)
  //     .attr('y', height / 2)
  //     .attr('dy', '.35em')
  //     .attr('text-anchor', 'start')
  //     .text(xScale.invert(sliderPosition).getFullYear());


  //   // 创建拖拽行为
  //   const drag = d3.drag<SVGGElement, unknown>()
  //     .on('drag', (event) => {
  //       let newX = Math.min(width, Math.max(0, event.x));
  //       setSliderPosition(newX);
  //       d3.select('.slider').attr('transform', `translate(${margin.left + newX},${margin.top})`);
  //       svg.select('.slider-text.left').text(xScale.invert(newX).getFullYear());
  //       svg.select('.slider-text.right').text(xScale.invert(newX).getFullYear());
  //     });

  //   slider.call(drag);
  // }, [events, windowWidth]);

  // useEffect(() => {
  //   // 当sliderPosition变化时，更新滑块位置，不需要重新绘制整个SVG
  //   const svg = d3.select(ref.current);
  //   const margin = { top: 20, right: 30, bottom: 20, left: 30 };

  //   svg.select('.slider').attr('transform', `translate(${margin.left + sliderPosition},${margin.top})`);
  //   svg.select('.slider-text.left').text(sliderPosition); // 更新文本可能需要更多的逻辑
  //   svg.select('.slider-text.right').text(sliderPosition); // 更新文本可能需要更多的逻辑
  // }, [sliderPosition]);


}
