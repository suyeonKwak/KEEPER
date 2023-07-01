import React, { useState, useEffect,useRef } from 'react';
import AWS from 'aws-sdk';
import Cookies from 'js-cookie';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import "./testpage.css";
import './testTarget.css';
const s3 = new AWS.S3({        
  accessKeyId: 'AKIA3QFVAHHVRIJK72AC',
  secretAccessKey: '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',
  region: 'ap-northeast-2',
});

function Target() {
  const videoRef = useRef(null);
  const Identifier = Cookies.get('identifier');
  const BucketName = 'coders-deep-fake';
  const FolderName = `${Identifier}/Target_Image/`;
  const URL = `${Identifier}/Target_Image/Target.png`;
  const navigate = useNavigate();

  const handleTrue=()=>{
    navigate("/Load");
  }
  const handleFalse=()=>{
    navigate("/Upload");

  }
 


  useEffect(() => {
    videoRef.current.play();
  }, []);
  function handleReset(event){
    navigate("/Upload");
  }
  const [imgUrl, setImgUrl] = useState(''); 
    useEffect(() => {
      const getImage = async () => {
        try {
          const data = await s3.listObjectsV2({ Bucket: BucketName, Prefix: FolderName }).promise();
          const imageKey = data.Contents.find(obj => obj.Key.endsWith('Target.png') && obj.Key.startsWith(FolderName));
          const url = s3.getSignedUrl('getObject', { Bucket: BucketName, Key: imageKey.Key });
          console.log(url);
          setImgUrl(url);
        } catch (err) {
          console.log(err);
        }
      }
      getImage();
      }, []);
    return (
      <div>
           <img id="logoimg" onClick={handleReset} src={`${process.env.PUBLIC_URL}/../img/logo.svg`} alt="로고이미지"/>
            <video ref={videoRef} loop muted playsInline className="page1video">
            <source src={`${process.env.PUBLIC_URL}/video/deepfake_dark.mp4`}
              type="video/mp4" />
            </video> 
            <img className="stepimg" src={`${process.env.PUBLIC_URL}/../img/step2.svg`} alt="스텝이미지"/>

            <div id="content" style={{ backgroundColor:"rgb(27, 27, 27,0.4)"}} >
              <p className='title'>IS THIS FACE RIGHT?</p>
              <div id="targetimg">
                {imgUrl && <img src={imgUrl} alt="타겟이미지" />}
              </div>
              <div id="targetbtn">
                <button className="btn" onClick={handleTrue}>YES </button><br></br>
                <button className="btn" id="btn2" onClick={handleFalse}>NO</button>
              </div>
        </div>
      </div>
    );
    }
    export default Target;