'use client'

import { useEffect, useRef, useState, createContext, useContext, MutableRefObject } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { Texture, TextureLoader, Sprite as ThreeSprite, Raycaster, Intersection, Vector2, Vector3, BoxGeometry } from "three";
import { OverallContext } from "./contents";
import { zip } from "d3";

interface SpriteProps {
  initialPosition: [number, number, number];
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
          spriteRef.current.scale.lerp(new Vector3(texture.image.width / 1000, texture.image.height / 1000, 1), 0.03);
          spriteRef.current.material.opacity += (1 - spriteRef.current.material.opacity) * 0.1;
        } else {
          spriteRef.current.scale.lerp(new Vector3(texture.image.width / 1500, texture.image.height / 1500, 1), 0.5);
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
          spriteRef.current.scale.lerp(new Vector3(texture.image.width / 1500, texture.image.height / 1500, 1), 0.05);
          spriteRef.current.material.opacity += (1 - spriteRef.current.material.opacity) * 0.01;
          spriteRef.current.position.lerp(new Vector3(finalPosition[0], finalPosition[1], finalPosition[2]), 0.01);
        }
      }
    }
  });

  // 设置 Sprite 的 scale 以保持原始图片比例
  useEffect(() => {
    if (texture) {
      spriteRef.current!.scale.set(texture.image.width / 1500, texture.image.height / 1500, 1);
    }
  }, [texture]);

  return (
    <sprite ref={spriteRef} position={initialPosition} userData={{ "charID": charID, "groupID": GroupID }}>
      <spriteMaterial attach="material" map={texture} transparent opacity={opacity} alphaTest={0.2} />
    </sprite>
  );
};

const SpriteController = () => {
  const overallStatus = useContext(OverallContext);
  const initCubeSide = overallStatus.current.initCubeSize;

  const sprites = [];
  sprites.push(...loadOneFace(CubeFace.Front, initCubeSide, 0, 0));
  sprites.push(...loadOneFace(CubeFace.Left, initCubeSide, 1, 16));
  sprites.push(...loadOneFace(CubeFace.Top, initCubeSide, 2, 32));
  sprites.push(...loadOneFace(CubeFace.Back, initCubeSide, 0, 48));
  sprites.push(...loadOneFace(CubeFace.Right, initCubeSide, 1, 64));
  sprites.push(...loadOneFace(CubeFace.Bottom, initCubeSide, 2, 80));

  return sprites;
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
              lastSelected.current = obj;
            } else {
              // 原始对象，保持

            }

          } else {
            // 进入选择状态
            overallStatus.current.rotationSpeed = 0.02;
            overallStatus.current.cubeScale = 1.5;
            overallStatus.current.selectedID = obj.userData["charID"];

            lastSelected.current = obj;

            // 避免不恰当的 Timeout
            clearTimeout(exitSelectTimeout.current!);
          }
        } else {
          if (lastSelected.current) {
            // 退出选择
            overallStatus.current.rotationSpeed = 0.1;

            exitSelectTimeout.current = setTimeout(() => {
              if (!lastSelected.current) {
                overallStatus.current.cubeScale = 1;
              }
            }, 2000);

            lastSelected.current = null;
            overallStatus.current.selectedID = -1;

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
          overallStatus.current.selectedGID = spriteCharGroups[overallStatus.current.selectedID];

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


const spriteLocations: [number, number][] = [
  // Charset A
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
  // Charset B
  [45.50, 23.39],
  [21.60, 158.05],
  [42.64, 263.11],
  [100.08, 79.15],
  [84.67, 116.63],
  [105.61, 157.18],
  [83.82, 253.65],
  [169.72, 20.49],
  [154.58, 154.36],
  [192.83, 150.20],
  [181.00, 253.87],
  [257.47, 43.55],
  [262.06, 88.44],
  [231.03, 186.97],
  [254.45, 214.79],
  [273.45, 263.11],
  // Charset C
  [25.75, 34.23],
  [44.17, 84.02],
  [44.17, 134.58],
  [68.82, 199.92],
  [79.23, 276.90],
  [130.42, 28.54],
  [112.54, 88.46],
  [60.43, 206.74],
  [142.13, 182.60],
  [207.00, 112.46],
  [179.37, 276.90],
  [250.08, 23.70],
  [220.08, 80.39],
  [266.65, 174.50],
  [271.21, 194.65],
  [237.65, 254.92],
];

const spriteGroupLocations: [number, number][] = [
  // Charset A
  [132.02, 59.75],
  [138.48, 92.30],
  [140.83, 166.79],
  [86.76, 179.06],
  [203.98, 193.07],
  [138.78, 160.69],
  [88.75, 158.83],
  [89.04, 186.80],
  [216.32, 176.58],
  [106.20, 108.64],
  [89.66, 150.74],
  [191.81, 125.14],
  [180.72, 128.17],
  [130.97, 234.61],
  [135.43, 254.17],
  [145.23, 68.29],
  // Charset B
  [140.86, 65.30],
  [163.14, 165.27],
  [193.76, 141.93],
  [132.79, 61.42],
  [92.13, 146.61],
  [92.13, 194.12],
  [213.47, 190.48],
  [116.69, 103.59],
  [125.67, 105.62],
  [88.08, 188.32],
  [211.78, 191.84],
  [93.98, 153.70],
  [148.06, 174.46],
  [137.92, 246.80],
  [191.51, 121.74],
  [142.37, 239.20],
  [150.27, 73.60],
  [197.47, 144.96],
  [89.66, 150.74],
  [216.32, 176.58],
  // Charset C
  [159.78, 99.47],
  [138.48, 92.30],
  [191.81, 125.14],
  [84.33, 189.52],
  [130.97, 234.61],
  [156.32, 172.34],
  [139.83, 228.08],
  [103.01, 159.18],
  [86.76, 179.06],
  [138.78, 160.69],
  [132.02, 59.75],
  [218.85, 199.70],
];

const spriteCharGroups: number[] = [
  // 0 ~ 15 Chars (Charset 1)
  0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1,
  // 16 ~ 31 Chars (Charset 2)
  2, 2, 2, 3, 3, 3, 2, 3, 2, 2, 3, 2, 3, 3, 3, 2,
  // 32 ~ 47 Chars (Charset 3)
  4, 4, 5, 5, 4, 5, 5, 4, 5, 4, 4, 4, 5, 5, 5, 4,

  // 48 ~ 63 Chars (Dup Charset 1)
  0 + 6, 0 + 6, 1 + 6, 0 + 6, 1 + 6, 0 + 6, 1 + 6, 1 + 6, 0 + 6, 1 + 6, 0 + 6, 0 + 6, 1 + 6, 0 + 6, 1 + 6, 1 + 6,
  // 64 ~ 79 Chars (Dup Charset 2)
  2 + 6, 2 + 6, 2 + 6, 3 + 6, 3 + 6, 3 + 6, 2 + 6, 3 + 6, 2 + 6, 2 + 6, 3 + 6, 2 + 6, 3 + 6, 3 + 6, 3 + 6, 2 + 6,
  // 80 ~ 95 Chars (Dup Charset 3)
  4 + 6, 4 + 6, 5 + 6, 5 + 6, 4 + 6, 5 + 6, 5 + 6, 4 + 6, 5 + 6, 4 + 6, 4 + 6, 4 + 6, 5 + 6, 5 + 6, 5 + 6, 4 + 6,
];

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
  // Dup
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

function loadOneFace(face: CubeFace, side: number, locIndex: number, startIndex: number): JSX.Element[] {
  const initLocations = spriteLocations.slice(locIndex * 16, locIndex * 16 + 16);
  const mappedInitLocations = mapToCubeFace(initLocations, face, side);

  const finalLocations = spriteGroupLocations.slice(locIndex * 16, locIndex * 16 + 16);
  const mappedFinalLocations = mapToCubeFace(finalLocations, CubeFace.Front, side);

  return zip(mappedInitLocations, mappedFinalLocations).map((locs, index) => {
    const imagePath = `/intro/${16 * locIndex + index + 1}.png`;
    const charID = startIndex + index;

    return <Sprite key={charID} initialPosition={locs[0]} finalPosition={locs[1]} imagePath={imagePath} opacity={0.6}
      charID={charID} GroupID={spriteCharGroups[charID]} />;
  });

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
