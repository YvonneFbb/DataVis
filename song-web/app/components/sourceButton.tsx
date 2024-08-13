'use client'

import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom';

interface PopupProps {
  onClose: () => void;
  isFadingOut?: boolean;
}

const SourcePopup: React.FC<PopupProps> = ({ onClose, isFadingOut }) => {
  return ReactDOM.createPortal(
    <div className={`popup ${isFadingOut ? 'fade-out' : ''}`}>
      <div className="popup-content">
        <button className="popup-close" type="button" data-dismiss="popup" aria-label="Close" onClick={onClose}>
          <span aria-hidden="true" className="popup-aria-hidden">&times;</span>
        </button>
        <h2>数据来源</h2>
        <h3>参考文献</h3>
        <p>[1]刘元堂.宋代版刻书法研究[D].南京艺术学院,2012.</p>
        <p style={{marginTop: "-5px"}}>[2]孙琬淑.比例规律与设计标准——宋刻本“一版双面”板框高广的尺度范式研究[J].装饰,2020,(06):76-80.</p>
        <h3 style={{marginTop: "50px"}}>参考资料库</h3>
        <p>上海图书馆资料库</p>
        <p style={{marginTop: "-5px"}}>康熙字典</p>
      </div>
    </div>,
    document.getElementById('source_popup') as HTMLElement
  );
};


export const SourceButton = () => {
  const [isPopupOpen, setIsPopupOpen] = useState(false);
  const [isFadingOut, setIsFadingOut] = useState(false);

  const openPopup = () => {
    setIsPopupOpen(true);
  };
  const closePopup = () => {
    setIsFadingOut(true);
    setTimeout(() => {
      setIsFadingOut(false);
      setIsPopupOpen(false);
    }, 550);
  };

  return (
    <>
      <button className="source-button source-button-text" onClick={openPopup}>数据来源</button>
      {isPopupOpen && <SourcePopup onClose={closePopup} isFadingOut={isFadingOut} />}
    </>
  );
}