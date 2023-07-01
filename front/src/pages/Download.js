import React, { useState, useEffect,useRef } from 'react';
import AWS from 'aws-sdk';
import Cookies from 'js-cookie';
import "./testpage.css";
import "./Download.css";
import axios from 'axios';
import { Navigate, useNavigate } from 'react-router-dom';

import './Download.css';
import "./testpage.css";
const s3 = new AWS.S3({
  accessKeyId: 'AKIA3QFVAHHVRIJK72AC',
  secretAccessKey: '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',
  region: 'ap-northeast-2',
});

function Download() {
  const videoRef = useRef(null);
  const Identifier = Cookies.get('identifier');
  const BucketName = 'coders-deep-fake';
  const navigate = useNavigate();
  const [videoUrl, setVideoUrl] = useState('');
  useEffect(() => { 

    const Videourl=async()=>{
      try {
        const params = {
          Bucket: BucketName,
          Key: `${Identifier}/Video/result.mp4`,
        };
        const url = await s3.getSignedUrlPromise('getObject', params);
        console.log(url);
        setVideoUrl(url);
      } catch (error) {
        console.error('Error fetching video URL:', error);
      }
    };
    Videourl();
  }, []);
  useEffect(() => {
    videoRef.current.play();
  }, []);

  const handleDownload = () => {
    const link = document.createElement('a');
    link.href = videoUrl;
    link.download = 'video.mp4';
    link.click();
  };
  function handleReset(event){
    navigate("/Upload");
  }
  return(
      <div>
           <img id="logoimg" onClick={handleReset} src={`${process.env.PUBLIC_URL}/../img/logo.svg`} alt="로고이미지"/>
          <video ref={videoRef} loop muted playsInline className="page1video">
          <source src={`${process.env.PUBLIC_URL}/video/deepfake_dark.mp4`}
            type="video/mp4" />
          </video> 
          <img className="stepimg" src={`${process.env.PUBLIC_URL}/../img/step2.svg`} alt="스텝이미지"/>
          <div id="content3">
            <p className='title'>DOWNLOAD YOUR VIDEO</p>
            <div id="video_container">
            <video src={videoUrl} controls />
            </div>
            <button className='btn'onClick={handleDownload}>DOWNLOAD</button>
          </div>          
      </div>
  );
}
export default Download;