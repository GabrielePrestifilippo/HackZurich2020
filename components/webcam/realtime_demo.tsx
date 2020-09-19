import React from 'react';
import {
  ActivityIndicator,
  Image,
  Button,
  StyleSheet,
  View,
  Text,
  Dimensions,
  TouchableHighlight
} from 'react-native';
import Svg, { Circle, Rect, G, Line} from 'react-native-svg';

import * as Permissions from 'expo-permissions';
import { Camera } from 'expo-camera';
import { ExpoWebGLRenderingContext } from 'expo-gl';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import {cameraWithTensors} from '@tensorflow/tfjs-react-native';
interface ScreenProps {
  returnToMain: () => void;
}

const {width, height}= Dimensions.get('window')
interface ScreenState {
  finished: any,
  log: any,
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
const litters = ['bottle', 'can', 'cup']
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
      finished:null,
      log: null,
      resultImage: null,
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

  async handleImageTensorReady(
    images: IterableIterator<tf.Tensor3D>,
    updatePreview: () => void, gl: ExpoWebGLRenderingContext) {
    if(this.state.finished!=null) {
      return
    }
    const loop = async () => {


      if(!AUTORENDER) {
        updatePreview();
      }

        if (this.state.cocossdModel != null && this.state.finished===null) {
          const imageTensor = images.next().value;
          const predictions = await this.state.cocossdModel.detect(imageTensor);
            const litterFound =
                predictions?.filter(({score, class: type})=>score>0.75 && litters.includes(type));

            if(litterFound[0]){
              this.setState({finished:litterFound[0].class})
            }
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

    const [cocossdModel] =
      await Promise.all([this.loadCocossdModel()]);
    this.setState({
      hasCameraPermission: status === 'granted',
      isLoading: false,
      cocossdModel,
    });
  }


  renderBox() {
    const {predictions} = this.state;
    if(predictions != null) {
      const bottleBox = predictions.map((prediction) => {

      if(!prediction.bbox[3]){
        return
      }
        const x = prediction.bbox[0];
        const y = prediction.bbox[1];
        const width = prediction.bbox[2];
        const height = prediction.bbox[3];

          return <Rect
              key={x+y}
              x={x}
              y={y}
              width={width}
              height={height}
              fill="rgb(0,0,255,0.2)"
              strokeWidth="1"
              stroke="rgb(0,0,0,0.01)"
          />
        });

      return <Svg height='100%' width='100%'
                  viewBox={`0 0 ${inputTensorWidth} ${inputTensorHeight}`}
                  scaleX={1}>
        {bottleBox}
      </Svg>;
    } else {
      return null;
    }
  }

  render() {
    const {isLoading, log, finished} = this.state;

    let textureDims: { width: number; height: number; };
        textureDims = {
          height: 1200,
          width: 1600,
      };

    const camView = <View style={styles.cameraContainer}>

      {!isLoading? <TensorCamera
        // Standard Camera props
        style={styles.camera}
        type={this.state.cameraType}
        zoom={0}
        flashMode={finished? 'auto':'torch'}
        // tensor related props
        cameraTextureHeight={textureDims.height}
        cameraTextureWidth={textureDims.width}
        resizeHeight={inputTensorHeight}
        resizeWidth={inputTensorWidth}
        resizeDepth={3}
        onReady={this.handleImageTensorReady}
        autorender={AUTORENDER}
      />: <View></View>}
        <View style={styles.modelResults}>
          {this.renderBox()}
        </View>
    </View>;

    if(isLoading){
      return (
          <View style={{
            backgroundColor:'#ececec',
            flex:1,
            flexDirection:'row',
            alignItems:'center',
            justifyContent:'center'}}>
          <View style={styles.loadingIndicator}>
            <ActivityIndicator size='large' color='#FF0266' />
          </View>
          </View>
      )
    }

    return (
      <View style={{width:'100%', backgroundColor:'white'}}>
        {finished != null ? <View style={{
          flexDirection:'row',
          alignItems:'center',
          justifyContent:'center',
          width:400,
          height:800,
        }}>
          <View style={{
            alignItems:'center',
            justifyContent:'center',
            alignSelf:'center',
            width:300,
            height:200,
            borderRadius:4,
            top: -50,
            padding:20,
            backgroundColor:'rgba(20,200,255,0.8)'}}>
            <Text style={{fontSize:22, color:'white', textAlign:'center'}}>Great! you found a {finished}</Text>
            <TouchableHighlight
                style={{
                  width:200,
                  height:50,
                  top:20,
              backgroundColor:'#7bec8e',
              borderRadius:2,
            }}
                onPress={() => this.setState({finished:null})}
            >
              <Text style={{
                textAlign:'center',
                color:'white',
                top:10,
                fontWeight: 'bold',
                fontSize:20
              }}>Continue</Text>
            </TouchableHighlight>
          </View>
        </View>:  null}
        {log?.length ?
        <Text style={{backgroundColor:'rgba(0,0,0,0.2)'}}>
          {log}
        </Text>: null}

        {camView}
      </View>
    );
  }

}

const styles = StyleSheet.create({
  loadingIndicator: {
    position: 'absolute',
    zIndex: 200,
    top:300,
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
    // top: 100,
    width: width,
    height: height,
    zIndex: 0,
  },
  modelResults: {
    position:'absolute',
    // top: 100,
    width: width,
    height: height,
    zIndex: 99,
  }
});
