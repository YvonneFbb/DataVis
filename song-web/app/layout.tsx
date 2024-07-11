import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Fbb's Font Vision",
  description: "Visualizing the evolution of Chinese fonts",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
