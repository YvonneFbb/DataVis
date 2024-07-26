'use client'

import { useEffect, useRef, useState, createContext, useContext, MutableRefObject } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { Texture, TextureLoader, Sprite as ThreeSprite, Raycaster, Intersection, Vector2, Vector3 } from "three";

interface CanvasStatus {
  isLoaded: boolean;
  isFinalSelected: boolean;
  rotationSpeed: number;
  initCubeSize: number;
  cubeScale: number;
  selectedID: number;
}

const CanvasContext = createContext<MutableRefObject<CanvasStatus>>({} as MutableRefObject<CanvasStatus>);

interface SpriteProps {
  initialPosition: [number, number, number];
  imagePath: string;
  opacity: number;
}

const Sprite: React.FC<SpriteProps> = ({ initialPosition, imagePath, opacity }) => {
  const texture = useLoader(TextureLoader, imagePath);
  const spriteRef = useRef<ThreeSprite>(null);
  const canvasStatus = useContext(CanvasContext);

  const targetPosition = useRef(new Vector3(0, 0, 0));
  const radius = (initialPosition[0] ** 2 + initialPosition[1] ** 2) ** 0.5;
  const angle = useRef(Math.atan2(initialPosition[1], initialPosition[0]));

  useFrame(({ camera }) => {
    if (spriteRef.current) {
      angle.current += 1 / 60 * canvasStatus.current.rotationSpeed;

      // 更新位置
      targetPosition.current.x = radius * Math.cos(angle.current) * canvasStatus.current.cubeScale;
      targetPosition.current.y = radius * Math.sin(angle.current) * canvasStatus.current.cubeScale;
      targetPosition.current.z = (initialPosition[2]) * canvasStatus.current.cubeScale;

      spriteRef.current.position.lerp(targetPosition.current, 0.1);

      // 应用新的位置
      spriteRef.current.lookAt(camera.position);
    }
  });

  // 设置 Sprite 的 scale 以保持原始图片比例
  useEffect(() => {
    if (texture) {
      spriteRef.current!.scale.set(texture.image.width / 1500, texture.image.height / 1500, 1);
    }
  }, [texture]);

  return (
    <sprite ref={spriteRef} position={initialPosition}>
      <spriteMaterial attach="material" map={texture} transparent opacity={opacity} alphaTest={0.2}/>
    </sprite>
  );
};


// 方向枚举
enum CubeFace {
  Front,
  Back,
  Left,
  Right,
  Top,
  Bottom
}

function mapToCubeFace(locations: [number, number][], face: CubeFace, side: number): [number, number, number][] {
  const maxDimension = 300;
  const halfSide = side / 2;
  const mappedLocations: [number, number, number][] = [];

  locations.forEach(location => {
    const [x2D, y2D] = location;

    // 将二维坐标从左上角原点转换为立方体面中心为原点
    const x = ((x2D / maxDimension) * side) - halfSide; // 现在范围是 [-side/2, side/2]
    const y = ((y2D / maxDimension) * side) - halfSide; // 现在范围是 [-side/2, side/2]

    let point3D: [number, number, number];

    switch (face) {
      case CubeFace.Front:
        point3D = [x, -y, halfSide];
        break;
      case CubeFace.Back:
        point3D = [-x, -y, -halfSide];
        break;
      case CubeFace.Left:
        point3D = [-halfSide, -y, x];
        break;
      case CubeFace.Right:
        point3D = [halfSide, -y, -x];
        break;
      case CubeFace.Top:
        point3D = [x, -halfSide, -y];
        break;
      case CubeFace.Bottom:
        point3D = [x, halfSide, y];
        break;
    }

    mappedLocations.push(point3D);
  });

  return mappedLocations;
}

function loadOneFace(face: CubeFace, side: number): JSX.Element[] {
  const locations = spriteLocations[0];
  const mappedLocations = mapToCubeFace(locations, face, side);

  return mappedLocations.map((location, index) => {
    const [x, y, z] = location;
    const imagePath = `/intro/${index + 1}.png`;

    return <Sprite key={index} initialPosition={[x, y, z]} imagePath={imagePath} opacity={0.6} />;
  });
}

const SpriteController = () => {
  const canvasStatus = useContext(CanvasContext);
  const initCubeSide = canvasStatus.current.initCubeSize;

  const sprites = loadOneFace(CubeFace.Front, initCubeSide);
  sprites.push(...loadOneFace(CubeFace.Back, initCubeSide));
  sprites.push(...loadOneFace(CubeFace.Left, initCubeSide));
  sprites.push(...loadOneFace(CubeFace.Right, initCubeSide));
  sprites.push(...loadOneFace(CubeFace.Top, initCubeSide));
  sprites.push(...loadOneFace(CubeFace.Bottom, initCubeSide));

  return sprites;
}

function getActualIntersect(intersects: Intersection[]): Intersection | null {
  for (const intersect of intersects) {
    const obj = intersect.object;
    const uv = intersect.uv;

    if (obj instanceof ThreeSprite && obj.material.map && uv) {
      const texture = obj.material.map as Texture;
      if (isTransparent(texture, uv)) {
        continue; // 透明区域，忽略此交点
      }
    }

    return intersect;
  }

  return null; // 未找到有效交点
}

function isTransparent(texture: Texture, uv: Vector2): boolean {
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');

  canvas.width = texture.image.width;
  canvas.height = texture.image.height;
  context?.drawImage(texture.image, 0, 0, texture.image.width, texture.image.height);

  const x = Math.floor(uv.x * texture.image.width);
  const y = Math.floor((1 - uv.y) * texture.image.height);
  const pixel = context?.getImageData(x, y, 1, 1).data;

  if (pixel) {
    return pixel[3] < 128; // Alpha 小于 128 认为是透明
  }

  return true; // 如果无法获取像素数据，默认为透明
}


const CameraController = () => {
  const { scene, camera } = useThree();
  const canvasStatus = useContext(CanvasContext);

  // For select
  const mouse = useRef(new Vector2());
  const raycaster = new Raycaster();

  const radius = 5; // 摄像机围绕原点的半径
  const targetPosition = useRef(new Vector3(0, radius, 0)); // 目标位置

  const lastSelected = useRef<ThreeSprite | null>(null);
  const exit_select_timeout = useRef<NodeJS.Timeout | null>(null);
  const steady_select_timeout = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const handleMouseMove = (event: MouseEvent) => {
      const { innerWidth, innerHeight } = window;
      // 将鼠标位置归一化到 [-1, 1]
      mouse.current.x = (event.clientX / innerWidth) * 2 - 1;
      mouse.current.y = -(event.clientY / innerHeight) * 2 + 1;

      // 选择判断
      raycaster.setFromCamera(mouse.current, camera);
      const intersects = raycaster.intersectObjects(scene.children);
      const intersect = getActualIntersect(intersects);
      if (intersect) {
        const obj = intersect!.object as ThreeSprite;

        if (lastSelected.current) {
          // 正在选择状态
          if (obj != lastSelected.current) {
            // 选择了新对象
            canvasStatus.current.selectedID = obj.id;

            lastSelected.current.material.opacity = 0.6;
            lastSelected.current = obj;
            obj.material.opacity = 1;

            // 切换选择
            clearTimeout(steady_select_timeout.current!);
            steady_select_timeout.current = setTimeout(() => {
              canvasStatus.current.isFinalSelected = true;
            }, 3000);
          } else {
            // 原始对象，保持
            
          }

        } else {
          // 进入选择状态
          canvasStatus.current.rotationSpeed = 0.02;
          canvasStatus.current.cubeScale = 1.5;
          canvasStatus.current.selectedID = obj.id;

          obj.material.opacity = 1;

          lastSelected.current = obj;

          // 避免不恰当的 Timeout
          clearTimeout(exit_select_timeout.current!);

          // 准备选择稳定
          steady_select_timeout.current = setTimeout(() => {
            canvasStatus.current.isFinalSelected = true
          }, 3000);
        }
      } else {
        if (lastSelected.current) {
          // 退出选择
          const obj = lastSelected.current;
          obj.material.opacity = 0.6;

          exit_select_timeout.current = setTimeout(() => {
            if (!lastSelected.current) {
              canvasStatus.current.rotationSpeed = 0.1;
              canvasStatus.current.cubeScale = 1;
              canvasStatus.current.selectedID = -1;
            }
          }, 2000)

          lastSelected.current = null;
        } else {
        }
      }
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


const spriteLocations: [number, number][][] = [
  // One Face
  [
    [38.17, 28.54],
    [75.03, 97.75],
    [21.96, 210.47],
    [38.17, 227.19],
    [143.23, 47.44],
    [167.96, 97.75],
    [117.17, 114.0],
    [191.05, 172.7],
    [150.00, 227.1],
    [91.91, 277.19],
    [260.05, 28.54],
    [265.46, 118.4],
    [250.08, 201.9],
    [225.03, 201.0],
    [206.77, 284.8],
    [276.11, 277.1],
  ],
]

export const IntroCanvas = () => {
  const canvasStatus = useRef<CanvasStatus>({
    isLoaded: false,
    isFinalSelected: false,
    rotationSpeed: 0.1,
    initCubeSize: 2.5,
    cubeScale: 1,
    selectedID: -1,
  });


  return (<CanvasContext.Provider value={canvasStatus}><Canvas
    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'black' }}
    camera={{ position: [5, 0, 0], up: [0, 0, 1] }}
  >
    <ambientLight intensity={0.3} />
    <pointLight position={[10, 10, 10]} />
    {/* <axesHelper args={[10]} /> */}
    <SpriteController />
    <CameraController />
  </Canvas></CanvasContext.Provider>
  )
}