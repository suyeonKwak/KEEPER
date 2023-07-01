import React, { useState, useEffect,useRef } from 'react';
import AWS from 'aws-sdk';
import Cookies from 'js-cookie';
import axios from 'axios';
import "./Load.css";
import "./testpage.css";
import { useNavigate } from 'react-router-dom';

function Load() {
    const videoRef1 = useRef(null);
    const videoRef2 = useRef(null);
    const Identifier = Cookies.get('identifier');
    const navigate = useNavigate();
    const [num, setNum] = useState(0);

    useEffect(() => {
      videoRef1.current.play();
      videoRef2.current.play(); 
    }, []);
    useEffect(()=>{
      console.log("요청");
        axios.post('http://127.0.0.1:8000/api/coders/target', { id:Identifier})
        .then(response => {
          console.log(response.data);
          navigate("/Virtual");
        })
        .catch(error => {
          console.error('Error uploading image:', error);
        });
    },[]);
    useEffect(() => {
        const interval = setInterval(() => {
          if (num < 99) {
            setNum(Num => Num + 4);
          }
          else{
            clearInterval(interval);
          }
        }, 600);
        return () => clearInterval(interval);
      });
    return(
        <div>
            <img id="logoimg" src={`${process.env.PUBLIC_URL}/../img/logo.svg`} alt="로고이미지"/>
            <video ref={videoRef1} loop muted playsInline className="page1video">
            <source src={`${process.env.PUBLIC_URL}/video/deepfake_dark.mp4`}
              type="video/mp4" />
            </video> 
            <img className="stepimg" src={`${process.env.PUBLIC_URL}/../img/step3.svg`} alt="스텝이미지"/>
            <div id='content2'>
                <div className="load">
                <video ref={videoRef2} loop muted playsInline className="video">
                <source src={`${process.env.PUBLIC_URL}/video/deepfake_dark.mp4`}
                type="video/mp4" />
                
                </video>
                <div id="p_text">
                    <p id="num">{num}%</p> 
                   <p className='text'>가상얼굴을 생성중이에요</p>
                   <p className='text'>생성이 완료될때까지 잠시만 기다려주세요</p>   
                </div>
                </div>
            </div>
        </div>
    )




}
export default Load;