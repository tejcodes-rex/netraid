# Third-party components

NetraID is released under the Apache License 2.0; see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

It bundles or depends on the components below. Every one is permissively licensed,
and none requires a commercial licence, a runtime fee or a per-device cost. Their
full licence texts ship with the packages themselves.

```
NetraID bundles or depends on the components below. Every one is permissively
licensed, and none requires a commercial licence, a runtime fee or a per-device
cost. Their licence texts ship with the packages themselves.

On-device models, bundled in app/assets/models:

  BlazeFace (short range)        Apache-2.0   Google MediaPipe
  FaceLandmarker                 Apache-2.0   Google MediaPipe
  MiniFASNet V2                  Apache-2.0   minivision-ai, Silent-Face-Anti-Spoofing
  MobileFaceNet (ArcFace)        MIT          InsightFace model zoo

Runtime dependencies:

  react-native                   MIT          Meta
  react-native-vision-camera     MIT          Marc Rousavy
  react-native-worklets-core     MIT          Marc Rousavy
  vision-camera-resize-plugin    MIT          Marc Rousavy
  react-native-fast-tflite       MIT          Marc Rousavy
  @op-engineering/op-sqlite      MIT          OP Engineering
  SQLCipher                      BSD-3-Clause Zetetic LLC
  react-native-keychain          MIT          Oblador
  @react-native-community/netinfo MIT         React Native Community
  @react-navigation/native       MIT          React Navigation
  react-native-safe-area-context MIT          Th3rd Wave
  react-native-uuid              MIT          Eugene Hauptmann

Model conversion toolchain, ml/scripts, not shipped in the app:

  PyTorch                        BSD-3-Clause
  ONNX                           Apache-2.0
  onnx2tf                        MIT
  TensorFlow / LiteRT            Apache-2.0

EdgeFace was evaluated for face recognition and rejected: it scores higher than
MobileFaceNet but its licence prohibits commercial use, which would have made
this solution unusable for a public deployment.
```
