'use client'

import { useEffect, useRef, useState, createContext, useContext, MutableRefObject } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { Texture, TextureLoader, Sprite as ThreeSprite, Raycaster, Intersection, Vector2, Vector3, BoxGeometry } from "three";
import { OverallContext } from "./contents";

interface SpriteProps {
  // x, y, z, scale, angle
  initialPosition: [number, number, number, number, number];
  finalPosition: [number, number, number];
  imagePath: string;
  opacity: number;
  charID: number;
  GroupID: number;
}

const Sprite: React.FC<SpriteProps> = ({ initialPosition, finalPosition, imagePath, opacity, charID, GroupID }) => {
  const overallStatus = useContext(OverallContext);

  const texture = useLoader(TextureLoader, imagePath);
  const spriteRef = useRef<ThreeSprite>(null);

  const targetPosition = useRef(new Vector3(0, 0, 0));
  const radius = (initialPosition[0] ** 2 + initialPosition[1] ** 2) ** 0.5;
  const angle = useRef(Math.atan2(initialPosition[1], initialPosition[0]));

  const finalSpeed = useRef(0.02);
  const finalScale = useRef(1.5);
  const finalSize = useRef(1);

  const reachFinal = useRef(false);
  const finalEnd = useRef(false);

  useFrame(({ camera }) => {
    if (spriteRef.current && !finalEnd.current) {
      spriteRef.current.lookAt(camera.position);
      if (!overallStatus.current.isFinalSelected) {
        angle.current += 1 / 60 * overallStatus.current.rotationSpeed;

        // 更新位置
        targetPosition.current.x = radius * Math.cos(angle.current) * overallStatus.current.cubeScale;
        targetPosition.current.y = radius * Math.sin(angle.current) * overallStatus.current.cubeScale;
        targetPosition.current.z = (initialPosition[2]) * overallStatus.current.cubeScale;
        spriteRef.current.position.lerp(targetPosition.current, 0.1);

        // 选择操作
        if (overallStatus.current.selectedID == charID) {
          spriteRef.current.scale.lerp(new Vector3(texture.image.width / 200, texture.image.height / 200, 1), 0.03);
          spriteRef.current.material.opacity += (1 - spriteRef.current.material.opacity) * 0.1;
        } else {
          spriteRef.current.scale.lerp(new Vector3(texture.image.width / 220 * initialPosition[3], texture.image.height / 220 * initialPosition[3], 1), 0.5);
          spriteRef.current.material.rotation += (initialPosition[4] / 180 * Math.PI - spriteRef.current.material.rotation) * 0.5;
          spriteRef.current.material.opacity += (opacity - spriteRef.current.material.opacity) * 0.5;
        }

        // 应用新的位置
      } else {
        if (!reachFinal.current) {
          reachFinal.current = true;
          setTimeout(() => {
            finalEnd.current = true;
          }, 1000 * 15);
        }
        if (overallStatus.current.selectedGID != GroupID) {
          // 不相关笔画
          finalSpeed.current += (2 - finalSpeed.current) * 0.02;
          finalScale.current += (5 - finalScale.current) * 0.02;
          finalSize.current += (0 - finalSize.current) * 0.04;

          angle.current += 1 / 60 * finalSpeed.current;

          // 更新位置
          targetPosition.current.x = radius * Math.cos(angle.current) * finalScale.current;
          targetPosition.current.y = radius * Math.sin(angle.current) * finalScale.current;
          targetPosition.current.z = (initialPosition[2]) * finalScale.current;

          spriteRef.current.scale.lerp(new Vector3(finalSize.current, finalSize.current, finalSize.current), 0.05);
          spriteRef.current.position.lerp(targetPosition.current, 0.05);
        } else {
          // 留存笔画
          spriteRef.current.scale.lerp(new Vector3(texture.image.width / 220, texture.image.height / 220, 1), 0.05);
          spriteRef.current.material.opacity += (1 - spriteRef.current.material.opacity) * 0.01;
          spriteRef.current.material.rotation += (0 - spriteRef.current.material.rotation) * 0.01;
          spriteRef.current.position.lerp(new Vector3(finalPosition[0], finalPosition[1], finalPosition[2]), 0.01);
        }
      }
    }
  });

  // 设置 Sprite 的 scale 以保持原始图片比例
  useEffect(() => {
    if (texture) {
      spriteRef.current!.scale.set(0, 0, 0);
    }
  }, [texture]);

  return (
    <sprite ref={spriteRef} rotation={[3, 3, 3]} position={[initialPosition[0], initialPosition[1], initialPosition[2]]} userData={{ "charID": charID, "groupID": GroupID }}>
      <spriteMaterial attach="material" map={texture} transparent opacity={opacity} />
    </sprite>
  );
};

const SpriteController = () => {
  const overallStatus = useContext(OverallContext);
  const initCubeEdge = overallStatus.current.initCubeSize;

  // const sprites = [];
  // sprites.push(...loadOneFace(CubeFace.Front, initCubeSide, 0, 0));
  // sprites.push(...loadOneFace(CubeFace.Left, initCubeSide, 1, 16));
  // sprites.push(...loadOneFace(CubeFace.Top, initCubeSide, 2, 32));
  // sprites.push(...loadOneFace(CubeFace.Back, initCubeSide, 0, 48));
  // sprites.push(...loadOneFace(CubeFace.Right, initCubeSide, 1, 64));
  // sprites.push(...loadOneFace(CubeFace.Bottom, initCubeSide, 2, 80));

  return loadInitChars(50, initCubeEdge);
}

const CameraController = () => {
  const { scene, camera, gl } = useThree();
  const overallStatus = useContext(OverallContext);
  const canvas = gl.domElement;

  // For select
  const mouse = useRef(new Vector2());
  const raycaster = new Raycaster();

  const radius = 5; // 摄像机围绕原点的半径
  const targetPosition = useRef(new Vector3(radius, 0, 0)); // 目标位置

  const lastSelected = useRef<ThreeSprite | null>(null);
  const exitSelectTimeout = useRef<NodeJS.Timeout | null>(null);

  const [selectProgress, setSelectProgress] = useState(0);

  const reachFinal = useRef(false);
  const finalEnd = useRef(false);

  useEffect(() => {
    camera.setViewOffset(canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);

    const handleMouseMove = (event: MouseEvent) => {
      if (event.target instanceof HTMLElement && !(event.target as HTMLElement).className.startsWith('intro')) {
        // 跳过 popup 的内容
        return;
      }

      const { innerWidth, innerHeight } = window;
      // 将鼠标位置归一化到 [-1, 1]
      mouse.current.x = (event.clientX / innerWidth) * 2 - 1;
      mouse.current.y = -(event.clientY / innerHeight) * 2 + 1;

      // 选择判断
      if (!overallStatus.current.isFinalSelected) {
        raycaster.setFromCamera(mouse.current, camera);
        const intersects = raycaster.intersectObjects(scene.children);
        const intersect = getActualIntersect(intersects);
        if (intersect) {
          const obj = intersect!.object as ThreeSprite;

          if (lastSelected.current) {
            // 正在选择状态
            if (obj != lastSelected.current) {
              // 选择了新对象
              overallStatus.current.selectedID = obj.userData["charID"];
              overallStatus.current.selectedGID = obj.userData["groupID"];
              lastSelected.current = obj;
            } else {
              // 原始对象，保持
            }

          } else {
            // 进入选择状态
            // overallStatus.current.rotationSpeed = 2;
            overallStatus.current.rotationSpeed = 0.00;
            overallStatus.current.cubeScale = 2;
            overallStatus.current.selectedID = obj.userData["charID"];
            overallStatus.current.selectedGID = obj.userData["groupID"];


            lastSelected.current = obj;

            // 避免不恰当的 Timeout
            clearTimeout(exitSelectTimeout.current!);
          }
        } else {
          if (lastSelected.current) {
            // 退出选择
            // overallStatus.current.rotationSpeed = 0.1;
            overallStatus.current.rotationSpeed = 0.0;

            exitSelectTimeout.current = setTimeout(() => {
              if (!lastSelected.current) {
                overallStatus.current.cubeScale = 1;
              }
            }, 2000);

            lastSelected.current = null;
            overallStatus.current.selectedID = -1;
            overallStatus.current.selectedGID = -1;
          } else {
            // 无选择
          }
        }
      }
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
    };
  }, []);

  // 更新选择进度
  useFrame(() => {
    if (!overallStatus.current.isFinalSelected) {
      if (overallStatus.current.selectedID != -1) {
        setSelectProgress(selectProgress >= 100 ? 100 : selectProgress + 1);
        if (selectProgress >= 100) {
          overallStatus.current.isFinalSelected = true;

          // 设置介绍文字
          const introRef = overallStatus.current.introRef.current!;
          const charDesc = charDescGroups[(overallStatus.current.selectedGID)];
          (introRef.querySelector('.intro-chardesc-title') as HTMLElement).innerText = charDesc.title;
          (introRef.querySelector('.intro-chardesc-subtitle') as HTMLElement).innerText = 'AD ' + charDesc.year;
          (introRef.querySelector('.intro-chardesc-caption') as HTMLElement).innerText = charDesc.caption;
          (introRef.querySelector('.intro-chardesc-description') as HTMLElement).innerText = charDesc.desc;
          setTimeout(() => {
            const buttonBox = introRef.querySelector('.intro-button-container') as HTMLElement;
            buttonBox.classList.remove('show');
          }, 1000 * 1);
          setTimeout(() => {
            const charBox = introRef.querySelector('.intro-chardesc-container') as HTMLElement;
            charBox.classList.add('show');
          }, 1000 * 2.5);
        }
      } else {
        setSelectProgress(selectProgress <= 0 ? 0 : selectProgress - 2);
      }
    }
  });

  const phi = useRef(0);
  const viewOffsetX = useRef(0);
  useFrame(() => {
    if (!finalEnd.current) {
      if (!overallStatus.current.isFinalSelected) {
        // 根据鼠标纵向位置调整摄像机的俯仰角
        phi.current = (mouse.current.y * 0.25 + 0.5) * Math.PI; // 从 0.25π 到 0.75π，即从稍微向下到稍微向上

        targetPosition.current.x = radius * Math.sin(phi.current);
        targetPosition.current.y = 0;
        targetPosition.current.z = radius * Math.cos(phi.current);

        // 使用 Lerp 实现平滑过渡
        camera.position.lerp(targetPosition.current, 0.005); // 0.01 是插值因子，决定了移动的速度和平滑程度
        camera.lookAt(0, 0, 0); // 摄像机始终朝向原点
      } else {
        if (!reachFinal.current) {
          reachFinal.current = true;
          setTimeout(() => {
            finalEnd.current = true;
          }, 1000 * 15);
        }
        phi.current += (Math.PI / 2 - phi.current) * 0.1;
        viewOffsetX.current += (400 - viewOffsetX.current) * 0.05;

        targetPosition.current.x = (radius + 2) * Math.cos(phi.current);
        targetPosition.current.y = 0;
        targetPosition.current.z = (radius + 2) * Math.sin(phi.current);

        // 使用 Lerp 实现平滑过渡
        camera.setViewOffset(canvas.width, canvas.height, viewOffsetX.current, 0, canvas.width, canvas.height);
        camera.position.lerp(targetPosition.current, 0.02);
        camera.up.lerp(new Vector3(0, 1, 0), 0.01);
        camera.lookAt(0, 0, 0);
      }
    }
  });


  return null;
};

// [Books, ID, locations, scale, angle]
const spriteInitLocations: [number, number, number, number, number, number][] = [
  [1, 1, 30.510, 13.864, 4.238 / 7.306, 9.544],
  [1, 2, 25.735, 26.474, 13.533 / 30.207, 356.104],
  [1, 2, 24.150, 27.390, 8.465 / 30.207, 0],
  [1, 2, 23.900, 27.390, 8.465 / 30.207, 0],
  [1, 2, 23.511, 26.950, 8.465 / 30.207, 0],
  [1, 2, 30.095, 37.143, 2.253 / 30.207, 80.599],
  [1, 2, 35.304, 16.480, 12.691 / 30.207, 357.182],
  [1, 2, 25.839, 17.190, 12.681 / 30.207, 354.908],
  [1, 3, 30.579, 35.909, 1.95 / 8.554, 314.78],
  [1, 4, 19.877, 16.958, 3.116 / 13.872, 0],
  [1, 5, 27.486, 30.740, 6.158 / 21.113, 0.738],
  [1, 6, 18.176, 18.897, 3.981 / 12.989, 22.509],
  [1, 7, 37.489, 25.882, 6.473 / 23.836, 220.512],

  [2, 1, 17.722, 20.207, 4.964 / 11.994, 38.151],
  [2, 1, 30.759, 36.749, 2.127 / 11.994, 71.8],
  [2, 2, 17.322, 22.590, 5.320 / 13.863, 35.22],
  [2, 3, 30.310, 24.019, 2.135 / 7.710, 0],
  [2, 4, 16.630, 22.227, 3.937 / 12.479, 211.681],
  [2, 5, 16.374, 39.487, 4.175 / 16.585, 324.689],
  [2, 6, 29.318, 26.305, 4.078 / 13.132, 0],
  [2, 7, 21.769, 37.048, 4.766 / 21.617, 251.8],

  [3, 1, 41.302, 15.747, 4.541 / 10.853, 176.104],
  [3, 2, 37.489, 25.268, 6.130 / 12.265, 176.104],
  [3, 2, 34.413, 25.712, 6.161 / 12.265, 171.743],
  [3, 3, 29.633, 40.015, 4.293 / 7.489, 0],
  [3, 4, 18.574, 38.604, 7.071 / 11.853, 14.075],
  [3, 5, 11.253, 41.250, 5.009 / 13.290, 333.502],
  [3, 5, 15.645, 39.329, 8.079 / 13.290, 341.025],
  [3, 6, 41.459, 18.143, 2.731 / 10.983, 213.346],
  [3, 7, 42.364, 17.807, 3.415 / 20.726, 333.567],

  [4, 1, 31.096, 12.435, 3.313 / 7.724, 13.19],
  [4, 1, 27.794, 44.876, 2.463 / 7.724, 354.392],
  [4, 2, 30.579, 24.872, 2.924 / 10.462, 68.447],
  [4, 2, 30.255, 23.818, 2.965 / 10.462, 255.745],
  [4, 2, 30.538, 31.077, 2.512 / 10.462, 73.979],
  [4, 2, 33.618, 31.146, 4.324 / 10.462, 294.819],
  [4, 2, 28.578, 46.157, 2.195 / 10.462, 283.819],
  [4, 3, 35.829, 32.692, 5.659 / 8.907, 47.959],
  [4, 4, 30.374, 26.516, 2.703 / 14.092, 68.045],
  [4, 5, 26.233, 31.077, 8.153 / 15.616, 352.394],
  [4, 6, 40.026, 25.712, 2.083 / 13.156, 352.967],
  [4, 7, 41.765, 36.515, 16.74 / 23.769, 0],

  [5, 1, 30.884, 11.328, 2.617 / 6.923, 0],
  [5, 1, 18.806, 26.422, 0.69 / 6.923, 239.93],
  [5, 2, 37.477, 25.475, 7.443 / 14.288, 350.205],
  [5, 3, 29.738, 39.487, 4.082 / 7.268, 0],
  [5, 4, 16.630, 23.840, 4.041 / 12.926, 17.318],
  [5, 5, 25.895, 31.867, 11.3 / 15.968, 356.104],
  [5, 5, 25.632, 30.740, 11.3 / 15.968, 356.104],
  [5, 6, 42.982, 16.679, 2.179 / 10.904, 23.243],
  [5, 7, 32.311, 29.923, 7.405 / 24.955, 178.679],

  [6, 1, 30.774, 11.772, 3.812 / 6.041, 341.8],
  [6, 2, 30.536, 39.107, 2.598 / 13.605, 70.807],
  [6, 2, 36.154, 16.480, 12.604 / 13.605, 346.208],
  [6, 3, 43.243, 17.614, 3.947 / 7.881, 224.775],
  [6, 4, 34.807, 26.134, 5.810 / 11.297, 334.174],
  [6, 4, 38.734, 25.882, 5.810 / 11.297, 334.174],
  [6, 4, 22.509, 17.928, 5.827 / 11.297, 335.893],
  [6, 4, 18.806, 20.655, 5.147 / 11.297, 27.048],
  [6, 4, 25.511, 17.814, 4.458 / 11.297, 335.893],
  // [6, 4, 4.458, 1.411, ]
  [6, 4, 30.084, 14.699, 2.821 / 11.297, 83.02],
  [6, 5, 13.844, 40.422, 4.889 / 14.972, 336.891],
  [6, 6, 18.875, 18.621, 3.749 / 11.294, 17.694],
  [6, 7, 40.551, 35.909, 12.431 / 19.342, 354.69],
]

// [locations]
const spriteFinalLocations: [number, number][] = [
  // 01 渭南文集
  [29.142, 11.913],
  [31.206, 17.302],
  [28.864, 32.740],
  [19.646, 29.648],
  [15.958, 35.424],
  [39.387, 26.134],
  [43.279, 37.294],
  // 02 长短经
  [28.653, 9.190],
  [25.621, 17.514],
  [28.638, 31.040],
  [19.687, 27.783],
  [17.586, 33.454],
  [36.509, 22.039],
  [40.700, 34.234],
  // 03 资治通鉴
  [27.335, 13.566],
  [24.269, 20.314],
  [28.515, 32.414],
  [19.195, 28.395],
  [19.196, 33.310],
  [38.555, 22.264],
  [43.465, 31.824],
  // 04 昆山杂咏
  [30.016, 10.903],
  [25.941, 17.889],
  [27.005, 31.335],
  [18.344, 28.847],
  [18.872, 33.778],
  [36.268, 22.545],
  [41.120, 34.483],
  // 05 尚书正义
  [26.716, 12.148],
  [25.077, 18.353],
  [28.711, 30.588],
  [18.000, 27.704],
  [18.428, 32.758],
  [37.380, 21.828],
  [43.456, 33.153],
  // 06 汉官仪
  [26.860, 12.152],
  [25.578, 18.133],
  [28.737, 31.141],
  [20.878, 27.025],
  [18.696, 33.527],
  [34.691, 24.195],
  [41.219, 32.650],
]

type CharDesc = {
  title: string;
  caption: string;
  year: number;
  desc: string;
  page: number;
}

export const charDescGroups: CharDesc[] = [
  { title: "《窦氏联珠集》", year: 1178, page: 7, caption: "略带褚遂良笔意", desc: "淳熙五年蕲州刻本《窦氏联珠集》略带褚遂良笔意，在宋代版刻楷书中别具一格。" },
  { title: "《昆山杂咏》", year: 1207, page: 8, caption: "近于行楷的字体", desc: "宋开禧三年昆山县斋刻本《昆山杂咏》以行楷笔意写稿，多有连笔及简写处。提按顿挫，笔意毕现。结字流美而富有新意，绝非俗手所书。" },
  { title: "《新定三礼图》", year: 1175, page: 7, caption: "欧颜型", desc: "到了南宋中期，近欧型依然盛行，其他三种类型则较为少见。欧颜型仅见于宋淳熙二年镇江府学刻公文纸印本《新定三礼图》一书，惜笔画细瘦，俊美有余，古意不足。" },
  { title: "《事类赋》", year: 1146, page: 6, caption: "近欧型", desc: "绍兴十六年两浙东路茶盐司刻本《事类赋》，半页八行，行十六至二十字不等。小字双行，行二十五至二十七字不等。白口，左右双边。书字体取法欧阳，字口清晰。" },
  { title: "《渭南文集》", year: 1220, page: 8, caption: "程式化", desc: "原本鲜活的欧字，逐步向程式化方向发展。这类风格占据了现存南宋中期浙本书籍的大多数，如：宋嘉定十三年陆子遹溧阳学宫刻本《渭南文集》。" },
  { title: "《窦氏联珠集》", year: 1178, page: 7, caption: "略带褚遂良笔意", desc: "淳熙五年蕲州刻本《窦氏联珠集》略带褚遂良笔意，在宋代版刻楷书中别具一格。" },
]


const Cube = () => {
  return (
    <lineSegments position={[0, 0, 0]}>
      <edgesGeometry args={[new BoxGeometry(2.5, 2.5, 2.5)]} />
      <lineBasicMaterial color="white" />
    </lineSegments>
  );
};

export const IntroCanvas = () => {
  return (<Canvas
    style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'black' }}
    camera={{ position: [5, 0, 0], up: [0, 0, 1] }}
    id="intro-canvas"
  >
    <ambientLight intensity={0.3} />
    <pointLight position={[10, 10, 10]} />
    {/* <axesHelper args={[10]} /> */}
    {/* <Cube /> */}
    <SpriteController />
    <CameraController />
  </Canvas>
  )
}

// 方向枚举
enum CubeFace {
  Front,
  Back,
  Left,
  Right,
  Top,
  Bottom
}

function loadInitChars(maxDimension: number, edge: number): JSX.Element[] {
  let sprites = []
  // 遍历 spriteInitLocations
  let key = 0;
  for (const loc of spriteInitLocations) {
    const [bookID, charID, initX2D, initY2D, scale, angle] = loc;
    const imagePath = `/intro/${(bookID - 1) * 7 + (charID)}.png`;

    const initX = ((initX2D / maxDimension) * edge) - edge / 2;
    const initY = ((initY2D / maxDimension) * edge) - edge / 2;
    const initialPosition: [number, number, number, number, number] = [0, initX, -initY, scale, angle];

    const finalTmp = spriteFinalLocations[(bookID - 1) * 7 + (charID - 1)];
    const finalX = ((finalTmp[0] / maxDimension) * edge) - edge / 2;
    const finalY = ((finalTmp[1] / maxDimension) * edge) - edge / 2;
    const finalPosition: [number, number, number] = [finalX, -finalY, 0];

    sprites.push(
      <Sprite key={key++} initialPosition={initialPosition} finalPosition={finalPosition} imagePath={imagePath} opacity={0.6}
        charID={key} GroupID={bookID} />
    )
  }

  return sprites;
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
