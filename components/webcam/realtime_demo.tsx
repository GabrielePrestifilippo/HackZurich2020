/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import React, {Fragment} from 'react';
import {ActivityIndicator, Button, StyleSheet, View, Platform, Text } from 'react-native';
import Svg, { Circle, Rect, G, Line} from 'react-native-svg';

import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import { ExpoWebGLRenderingContext } from 'expo-gl';

import * as tf from '@tensorflow/tfjs';
import * as blazeface from '@tensorflow-models/blazeface';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import {cameraWithTensors} from '@tensorflow/tfjs-react-native';
interface ScreenProps {
  returnToMain: () => void;
}

interface ScreenState {
  hasCameraPermission?: boolean;
  // tslint:disable-next-line: no-any
  cameraType: any;
  isLoading: boolean;
  cocossdModel?:any;
  predictions?: any;
  // tslint:disable-next-line: no-any
  faceDetector?: any;
  modelName: string;
}

const inputTensorWidth = 152;
const inputTensorHeight = 200;

const AUTORENDER = true;

// tslint:disable-next-line: variable-name
const TensorCamera = cameraWithTensors(Camera);

export class RealtimeDemo extends React.Component<ScreenProps,ScreenState> {
  rafID?: number;

  constructor(props: ScreenProps) {
    super(props);
    this.state = {
      isLoading: true,
      cameraType: Camera.Constants.Type.back,
      modelName: 'cocoSsd',
      predictions: null
    };
    this.handleImageTensorReady = this.handleImageTensorReady.bind(this);
  }

  async loadCocossdModel() {
    const model =  await cocoSsd.load({
      base: 'lite_mobilenet_v2'
    });
    return model;
  }

  async loadBlazefaceModel() {
    const model =  await blazeface.load();
    return model;
  }

  async handleImageTensorReady(
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void, gl: ExpoWebGLRenderingContext) {
    const loop = async () => {
      if(!AUTORENDER) {
        updatePreview();
      }

        if (this.state.cocossdModel != null) {
          const imageTensor = images.next().value;
          const predictions = await this.state.cocossdModel.detect(
            imageTensor );

          console.log('xxx predictions:', predictions);
          this.setState({predictions});
          tf.dispose([imageTensor]);
        }

      if(!AUTORENDER) {
        gl.endFrameEXP();
      }
      this.rafID = requestAnimationFrame(loop);
    };

    loop();
  }

  componentWillUnmount() {
    if(this.rafID) {
      cancelAnimationFrame(this.rafID);
    }
  }

  async componentDidMount() {
    const { status } = await Permissions.askAsync(Permissions.CAMERA);

    const [blazefaceModel, cocossdModel] =
      await Promise.all([this.loadBlazefaceModel(), this.loadCocossdModel()]);

    this.setState({
      hasCameraPermission: status === 'granted',
      isLoading: false,
      faceDetector: blazefaceModel,
      cocossdModel,
    });
  }


  renderBox() {
    const {predictions} = this.state;
    if(predictions != null) {
      const faceBoxes = predictions.map((prediction) => {

      if(!prediction.bbox[3]){
        return
      }
        const x = prediction.bbox[0];
        const y = prediction.bbox[1];
        const width = prediction.bbox[2];
        const height = prediction.bbox[3];

          return <Rect
              x={x}
              y={y}
              width={width}
              height={height}
              fill="rgb(0,0,255)"
              strokeWidth="3"
              stroke="rgb(0,0,0)"
          />
        });


      const flipHorizontal = Platform.OS === 'ios' ? 1 : -1;
      return <Svg height='100%' width='100%'
                  viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}
                  scaleX={1}>
        {faceBoxes}
      </Svg>;
    } else {
      return null;
    }
  }

  render() {
    const {isLoading, predictions} = this.state;

    // TODO File issue to be able get this from expo.
    // Caller will still need to account for orientation/phone rotation changes
    let textureDims: { width: number; height: number; };
    if (Platform.OS === 'ios') {
        textureDims = {
          height: 1920,
          width: 1080,
        };
      } else {
        textureDims = {
          height: 1200,
          width: 1600,
        };
      }

    const camView = <View style={styles.cameraContainer}>
      <TensorCamera
        // Standard Camera props
        style={styles.camera}
        type={this.state.cameraType}
        zoom={0}
        // tensor related props
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={inputTensorHeight}
        resizeWidth={inputTensorWidth}
        resizeDepth={3}
        onReady={this.handleImageTensorReady}
        autorender={AUTORENDER}
      />
      <View style={styles.modelResults}>
      {this.renderBox()}
      </View>
    </View>;

    return (
      <View style={{width:'100%'}}>
        <View style={styles.sectionContainer}>
          <Button
            onPress={this.props.returnToMain}
            title='Back'
          />
          {predictions?.map(({score, class: type})=>(
              <View key={score}>
                <Text  style={{color:'black'}}>Class: type</Text>
                <Text  style={{color:'black'}}>Score: score</Text>
              </View>
          ))}

        </View>
        {isLoading ? <View style={[styles.loadingIndicator]}>
          <ActivityIndicator size='large' color='#FF0266' />
        </View> : camView}


      </View>
    );
  }

}

const styles = StyleSheet.create({
  loadingIndicator: {
    position: 'absolute',
    top: 20,
    right: 20,
    zIndex: 200,
  },
  sectionContainer: {
    marginTop: 32,
    paddingHorizontal: 24,
  },
  cameraContainer: {
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
    height: '100%',
    backgroundColor: '#fff',
  },
  camera : {
    position:'absolute',
    left: 50,
    top: 100,
    width: 600/2,
    height: 800/2,
    zIndex: 1,
    borderWidth: 1,
    borderColor: 'black',
    borderRadius: 0,
  },
  modelResults: {
    position:'absolute',
    left: 50,
    top: 100,
    width: 600/2,
    height: 800/2,
    zIndex: 20,
    borderWidth: 1,
    borderColor: 'black',
    borderRadius: 0,
  }
});
