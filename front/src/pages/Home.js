import React, { useRef, useEffect } from "react";
import './Home.css';
import { useNavigate } from 'react-router-dom';
import Upload from './Upload';

function Home() {
  const videoRef = useRef(null);
  const navigate = useNavigate();

  const handleClick=()=>{ 
    videoRef.current.pause();
    navigate("/Upload");
  }
  return (
    <div className="background" onClick={handleClick}>
        <video ref={videoRef} autoPlay muted className="homevideo" >
        <source src={`${process.env.PUBLIC_URL}/video/deepfake_txt.mp4`}
        type="video/mp4"/>
        </video>
        <div id="next">
          <span id="start">Start<img className="nextIcon" alt="nextIcon" src={`${process.env.PUBLIC_URL}/img/double_down.svg`}></img></span> 
        </div>
    </div>
  );
}

export default Home;
