'use client'

import { useEffect, useRef, useState } from "react";
import { Canvas, useFrame, useLoader, useThree, extend } from "@react-three/fiber";
import { TextureLoader, Sprite as ThreeSprite, Vector3, PerspectiveCamera} from "three";

interface SpriteProps {
  initialPosition: [number, number, number];
  imagePath: string;
  opacity: number;
}

const Sprite: React.FC<SpriteProps> = ({ initialPosition, imagePath, opacity }) => {
  const texture = useLoader(TextureLoader, imagePath);
  const spriteRef = useRef<ThreeSprite>(null);
  const rotationSpeed = 0.3;

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
      spriteRef.current!.scale.set(texture.image.width / 1000, texture.image.height / 1000, 1);
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
  const radius = 4; // 摄像机围绕原点的半径

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
    // // 计算摄像机新的位置
    // const phi = (mouse.current.y * 0.5 + 0.5) * Math.PI; // 俯仰角，限制在 [0, π]
    // const theta = mouse.current.x * Math.PI + Math.PI; // 方位角，限制在 [0, 2π]

    // camera.position.x = radius * Math.sin(phi) * Math.cos(theta);
    // camera.position.y = radius * Math.cos(phi);
    // camera.position.z = radius * Math.sin(phi) * Math.sin(theta);

    camera.lookAt(0, 0, 0); // 摄像机始终朝向原点
  });

  return null;
};


export const IntroCanvas = () => {
  const numSprites = 8;  // 定义 sprite 数量
  const radius = 1;      // 定义 sprite 分布的半径

  return (<><Canvas
    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'black' }}
    camera={{ position: [0, 4, 0] }}
  >
    <ambientLight intensity={0.3} />
    <pointLight position={[10, 10, 10]} />
    <CameraController />
    {Array.from({ length: numSprites }, (_, index) => (
      <Sprite
        key={index}
        initialPosition={[
          radius * Math.cos((index / numSprites) * 2 * Math.PI),
          radius * Math.sin((index / numSprites) * 2 * Math.PI),
          0
        ]}
        imagePath={`/intro/${index + 1}.png`}
        opacity={0.75}
      />
    ))}
  </Canvas></>
  )
}