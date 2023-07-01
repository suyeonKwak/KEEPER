import React from 'react';
import { BrowserRouter,Routes,Route } from 'react-router-dom';
import Home from './pages/Home';
import Upload from './pages/Upload';
import Virtual from './pages/Virtual';
import Target from './pages/Target';
import Download from './pages/Download';
import Load from './pages/Load';
import Load_Video from './pages/Load_Video';
import './global.css'; 
import './App.css';

function App() {
  return (
    <BrowserRouter basename={process.env.PUBLIC_URL}>
    <div className="App">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/Upload" element={<Upload />} />
          <Route path="/Target" element={<Target />} />
          <Route path="/Virtual" element={<Virtual />} />
          <Route path="/Download" element={<Download />} />
          <Route path="/Load" element={<Load />} />
          <Route path="/Load" element={<Load />} />
          <Route path="/Load_Video" element={<Load_Video/>}/>
        </Routes>
    </div>
    </BrowserRouter>
    
  );
}

export default App;
