'use client'

import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';

interface PopupProps {
  onClose: () => void; // 关闭弹窗的函数
}

const SourcePopup: React.FC<PopupProps> = ({ onClose }) => {
  // 使用useEffect来处理挂载和卸载逻辑
  useEffect(() => {
    // 确保滚动被禁用，或者进行其他需要的效果
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, []);


  return ReactDOM.createPortal(
    <div className='popup'>
      <div className="popup__content">
        <button className="popup-close" type="button" data-dismiss="popup" aria-label="Close" onClick={onClose}>
          <span aria-hidden="true" className="popup-aria-hidden">&times;</span>
        </button>
        <h2 className="popup__title">数据来源</h2>
        <h3>参考文献：</h3>
        <ul>
          <li>刘元堂.宋代版刻书法研究[D].南京艺术学院,2012.</li>
        </ul>
      </div>
    </div>,
    document.getElementById('source_popup') as HTMLElement
  );
};


export const SourceButton = () => {
  const [isPopupOpen, setIsPopupOpen] = useState(false);

  const openPopup = () => setIsPopupOpen(true);
  const closePopup = () => setIsPopupOpen(false);

  return (
    <div>
      <button className="button button--clean" onClick={openPopup}>数据来源</button>
      {isPopupOpen && <SourcePopup onClose={closePopup} />}
    </div>
  );
}