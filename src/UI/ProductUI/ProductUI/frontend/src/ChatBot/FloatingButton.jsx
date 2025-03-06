import React from "react";
import "./FloatingButton.css";
import bot from '../../public/chatbott.png'; 

function FloatingButton({ setIsChatOpen, isChatOpen }) {
  return (
    <div
      className={`floating-button ${isChatOpen ? 'hidden' : ''}`}
      onClick={() => setIsChatOpen(prev => !prev)}
    >
      <img className="pngchat" src={bot} alt="Chatbot" />
    </div>
  );
}

export default FloatingButton;
