# Module 9 — 3D Reconstruction Interface & Visualization Pipeline

## 1. Overview & Purpose

Module 9 implements the **3D representation, reconstruction interface, 3D rendering, 360° rotating-view generation, OLED display interface, and trapezoidal-prism visualization** for PhotonShield AI.

It translates upstream sensor representations (from Module 4 Mamba-Hybrid, Module 7 Decision Layer, and Module 8 PINN + RL) into spatial 3D scenes, renders non-destructive 360° orbital view sequences, and formats multi-view layouts for transparent trapezoidal optical prisms.

```
             MODULE 4 / 7 / 8
                    │
                    ▼
             Scene Input
                    │
                    ▼
          3D Reconstruction API
                    │
                    ▼
                Scene3D
                    │
                    ▼
             Point Cloud / 3D
                    │
                    ▼
              3D Renderer
                    │
                    ▼
            Rotating 2D Views
                    │
                    ▼
                 OLED
                    │
                    ▼
          Trapezoidal Prism
                    │
                    ▼
        Pseudo-holographic display
```

---

## 2. 3D Representation & Point-Cloud Format

- **Primary Representation**: `PointCloud` $(N \times 3)$ or $(N \times 3+F)$
  - First three columns: spatial coordinates $(x, y, z)$.
  - Optional per-point channels: `intensities`, `confidences` ($[0.0, 1.0]$), `velocities`, `semantic_classes`.
- **Validation**: Rejects non-finite values (NaN / Inf) and malformed shapes.
- **Coordinate Convention**: Configurable metadata (default: `right_handed_z_up` where $x=\text{lateral}$, $y=\text{depth/range}$, $z=\text{vertical}$).

---

## 3. Modular 3D Reconstruction Interface

Since final real 3D ground-truth datasets for photonic radar are not yet defined, Module 9 provides an abstract interface:

```python
class ThreeDReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, scene_input: SceneInput) -> Scene3D:
        pass
```

- **`SyntheticGeometryReconstructor`**: Generates demo cube, sphere, and vehicle point-clouds. Labeled strictly as **SYNTHETIC VISUALIZATION DATA**.
- **`PassThroughReconstructor`**: Consumes pre-computed raw points or upstream representation into a `Scene3D`.
- **Future Integration**: A learned `RadarTo3DNeuralReconstructor` can be plugged in directly behind `ThreeDReconstructor` without modifying rendering, camera, or display layers.

---

## 4. Virtual Camera & 3D Renderer

- **`VirtualCamera`**: Configurable position, look-at target, FOV, near/far clipping planes, and resolution ($W \times H$).
- **`PointCloudRenderer`**: Fast pure-numpy rasterizer with z-buffering (depth buffering), point splatting, and customizable color modes (`depth`, `semantic_class`, `confidence`, `solid`).

---

## 5. Non-Destructive 360° Rotating-View Generation

- **`RotatingViewGenerator`**: Generates an ordered sequence of 2D views (e.g. $0^\circ, 15^\circ, 30^\circ, \dots, 345^\circ$).
- **Non-Destructive**: The camera orbits the geometric centroid of the scene. The underlying 3D point cloud is **never mutated** in place.

---

## 6. OLED Interface & Frame Streaming

- **`SimulatedDisplayBackend`**: Software simulation that captures rendered frames in memory for testing, headless rendering, and test suites.
- **`HardwareDisplayBackend`**: Clean interface contract for physical SPI/I2C OLED display drivers.
- **`FrameStream`**: Buffered queue with target FPS pacing, looping/one-shot playback, and an automatic frame-dropping policy on overflow.

---

## 7. Trapezoidal Prism Optical Visualization

- **`TrapezoidalPrismRenderer`**: Renders a 4-view composite canvas (Front, Back, Left, Right) positioned around a central cross.
- When placed beneath a 4-face transparent trapezoidal prism, each face reflects its respective view upwards to create an upright floating virtual 3D image.
- **Optical Disclaimer**: This system produces a **pseudo-holographic / Pepper's-Ghost-style multi-view visualization**, NOT a true volumetric hologram.

---

## 8. Export Utilities

- **Stanford `.ply`**: ASCII point clouds with optional confidence and class attributes.
- **Binary `.npy`**: Raw point arrays.
- **JSON Metadata**: Scene bounds, timestamps, coordinate frames, and object detections.
- **PNG Sequences**: Rendered 2D rotation frames.

---

## 9. Streaming & Strict Causality

- In streaming mode, rendering a frame at timestep $t$ depends **only** on scene input at or before $t$.
- No lookahead to future scenes $t+1 \dots T$.
