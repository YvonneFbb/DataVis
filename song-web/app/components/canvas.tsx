'use client'

import { useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { TextureLoader, Sprite as ThreeSprite, Raycaster, Vector2, Vector3 } from "three";

interface SpriteProps {
  initialPosition: [number, number, number];
  imagePath: string;
  opacity: number;
}

const Sprite: React.FC<SpriteProps> = ({ initialPosition, imagePath, opacity }) => {
  const texture = useLoader(TextureLoader, imagePath);
  const spriteRef = useRef<ThreeSprite>(null);
  const rotationSpeed = 0.1;

  // 维护当前位置状态
  const position = useRef(new Vector3(...initialPosition));

  useFrame(({ camera, clock }) => {
    if (spriteRef.current) {
      const elapsedTime = clock.getElapsedTime();
      const angle = elapsedTime * rotationSpeed;

      // 更新位置
      position.current.x = initialPosition[0] * Math.cos(angle) - initialPosition[1] * Math.sin(angle);
      position.current.y = initialPosition[0] * Math.sin(angle) + initialPosition[1] * Math.cos(angle);
      position.current.z = initialPosition[2];

      // 应用新的位置
      spriteRef.current.position.copy(position.current);
      spriteRef.current.lookAt(camera.position);
    }
  });

  // 设置 Sprite 的 scale 以保持原始图片比例
  useEffect(() => {
    if (texture) {
      spriteRef.current!.scale.set(texture.image.width / 1200, texture.image.height / 1200, 1);
    }
  }, [texture]);

  return (
    <sprite ref={spriteRef} position={initialPosition}>
      <spriteMaterial attach="material" map={texture} transparent opacity={opacity} />
    </sprite>
  );
};


const CameraController = () => {
  const { camera } = useThree();
  const mouse = useRef({ x: 0, y: 0 });
  const radius = 5; // 摄像机围绕原点的半径
  const targetPosition = useRef(new Vector3(0, radius, 0)); // 目标位置

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      const { innerWidth, innerHeight } = window;
      // 将鼠标位置归一化到 [-1, 1]
      mouse.current.x = (event.clientX / innerWidth) * 2 - 1;
      mouse.current.y = -(event.clientY / innerHeight) * 2 + 1;
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  useFrame(() => {
    // 根据鼠标纵向位置调整摄像机的俯仰角
    const phi = (mouse.current.y * 0.25 + 0.5) * Math.PI; // 从 0.25π 到 0.75π，即从稍微向下到稍微向上

    targetPosition.current.x = radius * Math.sin(phi);
    targetPosition.current.y = 0;
    targetPosition.current.z = radius * Math.cos(phi);

    // 使用 Lerp 实现平滑过渡
    camera.position.lerp(targetPosition.current, 0.005); // 0.01 是插值因子，决定了移动的速度和平滑程度

    camera.lookAt(0, 0, 0); // 摄像机始终朝向原点
  });

  return null;
};


export const IntroCanvas = () => {
  // const numSprites = 64; // 定义 sprite 数量
  // const baseRadius = 1.5;  // 基础半径
  // const variation = 0.5; // 半径变化量

  // // 生成球面上的点
  // const sprites = Array.from({ length: numSprites }, (_, index) => {
  //   const phi = Math.acos(-1 + 2 * Math.random()); // [-1, 1] 范围内随机
  //   const theta = 2 * Math.PI * Math.random(); // [0, 2π] 范围内随机
  //   const radius = baseRadius + Math.random() * variation; // 为每个 sprite 分配不同的半径

  //   const x = radius * Math.sin(phi) * Math.cos(theta);
  //   const y = radius * Math.sin(phi) * Math.sin(theta);
  //   const z = radius * Math.cos(phi);
  //   const imagePath = `/intro/${index + 1}.png`; // 假设图片存放在 public/intro 文件夹

  //   return <Sprite key={index} initialPosition={[x, y, z]} imagePath={imagePath} opacity={0.6} />;
  // });

  const numSprites = 64;
  const cubeSize = 3; // 立方体的边长，从 -1 到 1 的范围

  // 生成立方体内的点
  const sprites = Array.from({ length: numSprites }, (_, index) => {
    const face = Math.floor(Math.random() * 6);
    const x = (face === 0 ? 1 : face === 1 ? -1 : (Math.random() * 2 - 1)) * cubeSize / 2;
    const y = (face === 2 ? 1 : face === 3 ? -1 : (Math.random() * 2 - 1)) * cubeSize / 2;
    const z = (face === 4 ? 1 : face === 5 ? -1 : (Math.random() * 2 - 1)) * cubeSize / 2;
    const imagePath = `/intro/${index + 1}.png`; // 假设图片存放在 public/intro 文件夹

    return <Sprite key={index} initialPosition={[x, y, z]} imagePath={imagePath} opacity={0.6} />;
  });

  return (<><Canvas
    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'black' }}
    camera={{ position: [5, 0, 0], up: [0, 0, 1] }}
  >
    <ambientLight intensity={0.3} />
    <pointLight position={[10, 10, 10]} />
    {/* <axesHelper args={[10]} /> */}
    <CameraController />
    {sprites}
  </Canvas></>
  )
}