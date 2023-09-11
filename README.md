# Level Sets

This is a 1-semester project for the course "3D Computer Vision and Geometry".  
It showcases:
1. Using a mesh's Laplacian properties for shrinking/inflation of the mesh.
2. Using a mesh's Signed Distance Field (SDF) and its respective zero – level set for fast collision detection.

## Features

- Part A: Laplacian properties
    - Estimation of differential coordinates and visual representation of their values.
    - Mesh shrinking/smoothing via Taubin's method.
    - Mesh inflation via Taubin's method.
- Part B: Collision Detection
    - Calculation of the mesh's Signed Distance Field (SDF) and visual representation of its values.
    - Calculation of the mesh's Vector Force/Normal Field and visual representation of the vectors on the mesh.
    - Collision detection of the mesh with:
        a. a sphere,
        b. an axis-aligned box, and
        c. another mesh
    using:
        a. traditional means of collision detection (triangle intersections, Separating Axis Theorem, etc.), and
        b. the previously calculated SDF
    - Estimation of the collision response force and visual representation of the vectors.

## Screenshot

![levelsets](https://github.com/ChristoforosVlachos/level-sets/assets/96950242/1e56d678-0fed-4d64-af68-dbc42432c495)

## Try it!

If you are using a 64-bit Windows computer, you can download a pre-built version of the app on the releases section.

## Build from source

If your computer uses a different operating system (please note that this program has only been tested on 64-bit Windows) or you insist on building the app yourself, you may do so using CMake and Visual Studio.  
This app is dependent upon, [VVRFramework](https://bitbucket.org/vvr/vvrframework); you will have to build that as well from its repo on bitbucket (requires Qt5 to build).
