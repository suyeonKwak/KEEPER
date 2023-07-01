import React, { useState, useEffect,useRef } from 'react';
import AWS from 'aws-sdk';
import Cookies from 'js-cookie';
import axios from 'axios';
import "./Load.css";
import "./testpage.css";
import { useNavigate } from 'react-router-dom';

function Load_Video() {
    const videoRef1 = useRef(null);
    const videoRef2 =    useRef(null);
    const Identifier = Cookies.get('identifier');
    const navigate = useNavigate();
    const [num, setNum] = useState(0);
    const [text1, setText1] = useState('영상을 만드는 중이에요');
    const [text2, setText2] = useState('화면을 닫으면 작업이 중지돼요');
    useEffect(() => {
      videoRef1.current.play();
      videoRef2.current.play(); 
    }, []);
    useEffect(()=>{
      console.log("요청");
      console.log(Identifier);
        axios.post('http://127.0.0.1:8000/api/coders/video', { id:Identifier})
        .then(response => {
          console.log("video endpoint return")
          console.log(response.data);
          setText1('영상이 거의 다 만들어졌어요');
          setText2('잠시만 기다려주세요');
          navigate("/Download");
        })
        .catch(error => {
          console.error('Error uploading image:', error);
        });
    },[]);
    useEffect(() => {
        const interval = setInterval(() => {
          if (num < 99) {
            setNum(Num => Num + 2);
          }
          else{
            clearInterval(interval);
          }
        }, 1000);
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
                <div id="p_text2">
                    <p id="num">{num}%</p> 
                   <p className='text'>{text1}</p>
                   <p className='text'>{text2}</p>   
                </div>
                </div>
            </div>
        </div>
    )




}
export default Load_Video;