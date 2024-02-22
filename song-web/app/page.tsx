import { Header } from "./components/header";
import { Footer } from "./components/footer";
import { Intro } from "./components/intro";

export default function Home() {
  return (
    <body id="body" className="visual-intro-active key-event">
      <Header />
      <Intro />
      <div id="portal-root"></div>
    </body>
  );
}
