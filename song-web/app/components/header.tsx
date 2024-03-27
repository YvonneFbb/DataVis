'use client'

import { SourceButton } from "./sourceButton";

export function Header() {
  return (
    <header id="header" className="header">
      <div className="header__button-container">
        <SourceButton />
      </div>
      <div className="header__logo-container">
        <div className="logo"></div>
      </div>
    </header>
  );
}