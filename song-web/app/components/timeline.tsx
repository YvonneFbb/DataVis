import React, { useRef, useEffect, useState } from 'react';
import * as d3 from 'd3';

export enum EventType {
  Empire,
  Year,
  Artist,
  Ou, Yan, Liu, Zhao, Others, Overview
};

export type TimelineEvent = {
  name: string;
  year: number;
  ty: EventType;
};

type TimelineProps = {
  events: TimelineEvent[];
  span: [number, number];
};

export function Timeline({ events, span }: TimelineProps) {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [windowHeight, setWindowHeight] = useState(window.innerHeight);
  const ref = useRef<SVGSVGElement>(null);

  // Update window width and height
  useEffect(() => {
    const handleResize = () => { setWindowWidth(window.innerWidth); setWindowHeight(window.innerHeight); };
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

    const sliderWidth = windowWidth / 6;
    const sliderHeight = viewHeight - 55;

    const attachDistance = 20;

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

    const eventsGroup = viewG.append('g')
      .attr('class', 'events')
      .attr('pointer-events', 'auto');

    const xScale = d3.scaleTime()
      .domain(span.map(date => new Date(date)))
      .range([0, windowWidth]);

    events.forEach(event => {
      const xPos = xScale(event.year);
      eventsGroup.append('circle')
        .attr('cx', xPos)
        .attr('cy', sliderHeight / 2)
        .attr('r', 5)
        .attr('style', 'cursor: pointer;')
        .attr('fill', '#000')
        .on('click', (event, d) => {
          const posX = event.x - sliderWidth / 2;
          slider.attr('x', posX);
          sliderRect.transition().duration(300).attr('transform', `translate(${posX}, 0)`);
          linesGroup.transition().duration(300).attr('transform', `translate(${posX + sliderWidth / 2}, ${sliderHeight / 2})`);
        })
    });

    const slider = viewG.append('g')
      .attr('class', 'slider');

    const sliderRect = slider.append('rect')
      .attr('class', 'box')
      .attr('transform', `translate(0, 0)`)
      .attr('width', `${sliderWidth}`)
      .attr('height', `${sliderHeight}`)
      .attr('style', 'cursor: move;');

    // Add lines
    const linesGroup = slider.append('g')
      .attr('class', 'lines-decor')
      .attr('transform', `translate(${sliderWidth / 2}, ${sliderHeight / 2})`);

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

    let dragOffset = 0;

    const drag = d3.drag()
      .on('start', function (event, d) {
        const currentX = d3.select(this).attr('x');
        dragOffset = event.x - (currentX as unknown as number);
        d3.select(this).classed('active', true);
      })
      .on('drag', function (event, d) {
        let newX = event.x - dragOffset;
        newX = Math.max(0, Math.min(windowWidth - sliderWidth, newX));

        d3.select(this).attr('x', newX);
        sliderRect.attr('transform', `translate(${newX}, 0)`);
        linesGroup.attr('transform', `translate(${newX + sliderWidth / 2}, ${sliderHeight / 2})`);
      })
      .on('end', function (event, d) {
        d3.select(this).classed('active', false);
        const currentX = (+ d3.select(this).attr('x') as unknown as number) + sliderWidth / 2;

        let closestDistance = Infinity;
        let closestEventX = null;

        events.forEach(event => {
          const eventX = xScale(event.year);
          const distance = Math.abs(eventX - currentX);
          if (distance < attachDistance && distance < closestDistance) {
            closestDistance = distance;
            closestEventX = eventX - sliderWidth / 2;
          }
        });

        if (closestEventX !== null) {
          d3.select(this).attr('x', closestEventX);
          sliderRect.transition().duration(300).attr('transform', `translate(${closestEventX}, 0)`);
          linesGroup.transition().duration(300).attr('transform', `translate(${closestEventX + sliderWidth / 2}, ${sliderHeight / 2})`);
        }
      });

    slider.call(drag as any);
  }, [events, windowWidth, windowHeight])

  return (
    <svg ref={ref} width={windowWidth} height={windowHeight} />
  );
}
