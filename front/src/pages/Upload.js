import React, { useState ,useEffect,useRef} from 'react';
import AWS from 'aws-sdk';
import { v4 as uuidv4 } from 'uuid';
import "./testUpload.css"
import Cookies from 'js-cookie';
import { useNavigate } from 'react-router-dom';
import "./testpage.css";
import axios from 'axios';

//export const accessKeyId = process.env.REACT_APP_AWS_ACCESS_KEY_ID;
//export const secretAccessKey = process.env.REACT_APP_AWS_SECRET_ACCESS_KEY;
function Upload() {
  const videoRef1 = useRef(null); 
  const videoRef2 = useRef(null);   
  useEffect(() => {
    videoRef1.current.play();
    videoRef2.current.play(); 
  }, []);

  const [fileInfo,setFileInfo]=useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [identifier, setIdentifier] = useState('');
  const [isLoading, setIsLoading] = useState(false); // 로딩 상태 추가
  const [isNext, setIsNext] = useState(false);
  const navigate = useNavigate();
  const buttonRef = useRef(null); // 버튼 ref 생성
  const [num, setNum] = useState(0);

  let Identifier ='';

  function handleDragover(event){
    event.preventDefault();
  }
  function handleDrop(event){
    event.preventDefault(); 
    setSelectedFile(event.dataTransfer.files[0]);
    setButtonText('Upload');
    setFileInfo(
      `파일 이름 : ${event.dataTransfer.files[0].name}, 
      파일 크기 : ${(event.dataTransfer.files[0].size / (1024*1024)).toFixed(2)} MB`
      );
  }
  
  function handleFileChange(event){
    event.preventDefault();
    setSelectedFile(event.target.files[0]);
    setButtonText('Upload');
    setFileInfo(
      `파일 이름 : ${event.target.files[0].name}, 
      파일 크기 : ${(event.target.files[0].size / (1024*1024)).toFixed(2)} MB`
      );
  }

  function handleUpload(event){
    if(selectedFile){
      
    if (buttonRef.current) {
      console.log(buttonRef);
        buttonRef.current.classList.add("loading");
        console.log(buttonRef);

    }
      const newIdentifier = uuidv4(); // 새로운 identifier 생성
      Cookies.set('identifier', newIdentifier);
      Identifier = Cookies.get('identifier');
      const s3 = new AWS.S3({
        accessKeyId: 'AKIA3QFVAHHVRIJK72AC',
        secretAccessKey: 'IHSJityy4mXQYMwLwnWLYpXcaXTRmpJ/qHz0tVuf',
        region: 'ap-northeast-2'
      });
      const params = {
        Bucket: 'coders-deep-fake',
        Key: `${newIdentifier}/video.mp4`,
        Body: selectedFile
      };
      s3.upload(params,function(err,data){
        if(err){
          console.error(err);
        }
        else{
          setIsLoading(true);
          console.log("식별자"+Identifier);
          setButtonText('취소');  
         
          console.log(Identifier);
          axios.post('http://127.0.0.1:8000/api/coders/Upload', { id: Identifier })
            .then(function(response) {
              Identifier=response.data.message;
              console.log(Identifier);
              setIsNext(true);
              document.getElementById('fileimg').src = `${process.env.PUBLIC_URL}/../img/done.svg`;
              document.getElementsByTagName('p')[0].textContent='분석이 완료되었습니다.';
              setButtonText('Next');
            })
            .catch(error => {
              console.log("error_upload");

            })



        }
      });

    }
  }
  function handleNextClick(event){
    navigate("/Target");

  }
  function handleReset(event){
    window.location.reload();
  }
  const [buttonText, setButtonText] = useState("분석 하기");
      return(
        <div>
          <img id="logoimg" src={`${process.env.PUBLIC_URL}/../img/logo.svg`} alt="로고이미지"/>
            <video ref={videoRef1} loop muted playsInline className="page1video">
            <source src={`${process.env.PUBLIC_URL}/video/deepfake_dark.mp4`}
              type="video/mp4" />
            </video> 
            <div id="main">
            <img className="stepimg" src={`${process.env.PUBLIC_URL}/../img/step1.svg`} alt="스텝이미지"/>
              <div id="content" >
              <video ref={videoRef2} loop muted playsInline className="video">
                <source src={`${process.env.PUBLIC_URL}/video/deepfake_dark.mp4`}
                type="video/mp4" />
                
                </video>
                {!selectedFile &&(
                  /*1.비디오 탐색*/ 
                  <label id="upload_box"  onDragOver={handleDragover} onDrop={handleDrop} style={{cursor:"pointer"}}>
                      <p className='title_upload' >UPLOAD YOUR VIDEO</p>
                      <div>
                        <img id="fileimg" src={`${process.env.PUBLIC_URL}/../img/file.svg`} alt="파일이미지"/>
                        <p id="upload_p1"></p>
                        <p id="upload_p2">(100MB이하, 1분이내)</p>
                        <input type="file" onChange={handleFileChange} style={{display:'none'}}/>
                      </div>
                  </label>
                )}
                {selectedFile &&(
                  <label id="upload_box" onDragOver={handleDragover} onDrop={handleDrop} >
                     {isLoading ? ( 
                      <div>
                        {isNext ? (
                          /*4.Target페이지 이동 */ 
                         <div>
                         <p className='title_upload'>CLICK THE NEXT BUTTON</p>
                          <img id="fileimg" src={`${process.env.PUBLIC_URL}/../img/done.svg`} alt="Next이미지"/>
                          <p id="upload_p2">{selectedFile.name}({(selectedFile.size/(1024*1024)).toFixed(2)}MB)</p>
                          <button className='btn' onClick={handleNextClick}>{buttonText}</button>
                         </div>
                        ):(
                          /* 3.영상 분석 */ 
                          <div>
                           <p className='title_upload' >ANALYZING THE VIDEO</p>   
                          <div className="spinner-container"id="fileimg">
                            <div className="spinner"></div>
                          </div>
                          <p id="upload_p2"> 영상을 분석중이에요.  ( 예상소요시간 : 30초 ) </p>
                          <button className='btn' onClick={handleReset}>{buttonText}</button>
                        </div>
                        )}
                      </div>
                     
                      ) : (
                        /* 2.업로드 버튼 클릭 */ 
                        <div>
                        <p className='title_upload'>UPLOAD YOUR VIDEO  </p>
                        <img id="fileimg" src={`${process.env.PUBLIC_URL}/../img/start.svg`} alt="시작이미지"/>
                        <p id="upload_p2">{selectedFile.name}({(selectedFile.size/(1024*1024)).toFixed(2)}MB)</p>
                        <button className="btn" id="uploadfile"  ref={buttonRef}  onClick={handleUpload}>{buttonText}</button>
                        </div>
                    )} 

                                      
                  </label>
                  
                )}


              </div>
        </div>
      </div>

    );
    
}

export default Upload;
