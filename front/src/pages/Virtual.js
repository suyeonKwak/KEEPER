import React, { useState, useEffect,useRef } from 'react';
import AWS from 'aws-sdk';
import Cookies from 'js-cookie';
import axios from 'axios';
import "./Virtual.css";
import "./testpage2.css";
import { useNavigate } from 'react-router-dom';

function Virtual() {
  const videoRef = useRef(null);
  const [images, setImages] = useState([]);
  const bucketName = 'coders-deep-fake';
  const FolderName = 'Virtual/';
  const [selectedImage, setSelectedImage] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
      videoRef.current.play();
  }, []);
  useEffect(() => {
    if (selectedImage) {
      document.getElementById('selected_img').innerHTML = `<img src="https://${bucketName}.s3.amazonaws.com/${selectedImage}" alt="Selected Image" />`;
    } else {
      document.getElementById('selected_img').innerHTML = `<img src="${process.env.PUBLIC_URL}/../img/test.svg" alt="test이미지"/>`;
    }
    console.log(selectedImage);
  }, [selectedImage]);
  
  useEffect(()=>{
    
    const s3 = new AWS.S3({
      accessKeyId: 'AKIA3QFVAHHVRIJK72AC',
      secretAccessKey: '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',
      region: 'ap-northeast-2',
    });

    const listObjects = async () => {
      try {
        const response = await s3.listObjectsV2({ Bucket: bucketName, Prefix: `${Identifier}/Virtual/` }).promise();
        const imageList = response.Contents.map(obj => obj.Key);

        setImages(imageList);
      } catch (error) {
        console.error('Error listing objects from S3:', error);
      }
  };
  listObjects();
  const interval = setInterval(() => {
    listObjects();
  }, 5000); 

  return () => {
    clearInterval(interval);
  };
  },[]);
  const Identifier = Cookies.get('identifier');
  const URL = `${selectedImage}`
  console.log(`https://${bucketName}.s3.ap-northeast-2.amazonaws.com/${Identifier}/Virtual/`);
  const handleStartClick = () => {
    console.log(URL);
    navigate("/Load_Video");
    if (selectedImage) {
      axios.post('http://127.0.0.1:8000/api/coders/virtual', { imageUrl: URL})
        .then(response => {
          console.log(response.data);

        })
        .catch(error => {
          console.error('Error uploading image:', error);
        });
    }
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
            <img className="stepimg" src={`${process.env.PUBLIC_URL}/../img/step3.svg`} alt="스텝이미지"/>
            <div id="content2" style={{ backgroundColor:"rgb(27, 27, 27,0.4)"}} >
              <p className="title">CHOOSE THE VIRTUAL FACE</p>
              <div id="selected_img">             

              </div>
              <button className='btn2'  onClick={handleStartClick}>Start</button>
              <div id="img_container">
                {images 
                  .filter(image => image.toLowerCase().endsWith('.png'))
                  .map((image, index) => (
                    <img 
                    key={index} 
                    src={`https://${bucketName}.s3.ap-northeast-2.amazonaws.com/${image}`} 
                    alt={`Image ${image}`} 
                    onClick={() => setSelectedImage(image)}
                    />
                ))}
              </div>

            </div>

        </div>
    );
}
export default Virtual;