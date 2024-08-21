import React, { useState,useEffect} from "react";
import './App.css';
import {Col,Row,Button,Form,Input,Select,checkbox,Radio,DatePicker, InputNumber,Modal} from 'antd';
import { useForm } from "antd/es/form/Form";
import axios from "axios";
import { ToastContainer } from "react-toastify";
import { toast } from "react-toastify";
import 'react-toastify/dist/ReactToastify.css';



function App() {

  const [isModalVisible, setIsModalVisible] = useState(false); 
  const [FormInsert] = useForm();
  const [speech,setSpeech]=useState("");
  const handleSubmit = () =>{
    let formData = FormInsert.getFieldValue();
    console.log(JSON.stringify(formData));
    axios.post('http://localhost:5000/add', formData)
    .then((response) => {
      setSpeech(response.data.Speech);
      const value = new SpeechSynthesisUtterance(speech);
      const voices = speechSynthesis.getVoices().filter(voice => voice.voiceURI === 'Google US English');
        if (voices.length > 0) {
          value.voice = voices[0];
        }
      window.speechSynthesis.speak(value);
      toast('Data saved', {
        position: "top-right",
        autoClose: 1000,
        hideProgressBar: false,
        closeOnClick: true,
        pauseOnHover: true,
        draggable: true,
        progress: undefined,
        theme: "light",
      });
      setIsModalVisible(true); 
    })
    .catch(function (error) {
      console.error('Error:', error);
    });
  }
  return (
    <div className="App" style={{position: "fixed",width: "100%", height: "100%", backgroundImage: 'url(/HA1.png)', backgroundSize: 'cover'}}>
      <header className="App-header">{/*
        <img src="https://upload.wikimedia.org/wikipedia/commons/4/47/Oshhf5T5_400x400.jpg" className="App-logo" alt="logo" />
        */} <p>
          HEALTH ADVISOR
        </p>
      </header>
    <bg-image src="https://advisorhealthcare.com/wp-content/uploads/2018/05/hero2.jpg">
    </bg-image>
       
    <Form
        labelCol={{ span: 14 }}
        wrapperCol={{ span: 14 }}
        layout="horizontal"
        onFinish={handleSubmit}
        style={{ maxWidth: 1000 }}
        form={FormInsert}
        className="form"
    >
      <Row gutter={[16,24]}>
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Date of Birth"
                name="Date of Birth"
                rules={[
                  {
                    required: true,
                  },
                ]}
              >
               
                <DatePicker
                  type="DOB"
                  style={{ width: '120%',border: "1px solid gold"}}
                />
               
               
      </Form.Item>
        </Col>
     
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Gender"
                name="Gender"
                rules={[
                  {
                    required: true,
                    message: 'Gender',
                  },
                ]}
      >
      <Radio.Group>
            <Radio value="Male" className="hover-color"> Male </Radio>
            <Radio value="Female" className="hover-color"> Female </Radio>
      </Radio.Group>
               
    </Form.Item>
        </Col>
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Height"
                name="Height"
                rules={[
                  {
                    required: true,
                    message: 'Height(cm)',
                  },
                ]}
              >
                <InputNumber
                  type="Height"
                  style={{border: "1px solid gold"}}
                />
    </Form.Item>
          </Col>
 
          <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Weight"
                name="Weight"
                rules={[
                  {
                    required: true,
                    message: 'Weight(cm)',
                  },
                ]}
              >
                <InputNumber
                  type="Weight"
                  style={{border: "1px solid gold"}}
                />
    </Form.Item>
        </Col>
 
 
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Physically Active"
                name="active"
                rules={[
                  {
                    required: true,
                  },
                ]}
    >
    <Radio.Group>
            <Radio value="Yes" className="hover-color"> Yes </Radio>
            <Radio value="No" className="hover-color"> No </Radio>
    </Radio.Group>
 
    </Form.Item>
        </Col>
 
 
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Active Smoker"
                name="Smoke"
                rules={[
                  {
                    required: true,
                  },
                ]}
    >
    <Radio.Group>
            <Radio value="Yes" className="hover-color"> Yes </Radio><p>Yes</p>
            <Radio value="No" className="hover-color"> No </Radio>
    </Radio.Group>
 
    </Form.Item>
        </Col>
 
 
 
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Systolic BP"
                name="Systolic_BP"
                rules={[
                  {
                    required: true,
                  },
                ]}
              >
                <InputNumber
                  type="Systolic_BP"
                  style={{border: "1px solid gold"}}
                />
    </Form.Item>
        </Col>
 
 
 
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Diastolic BP"
                name="Diastolic_BP"
                rules={[
                  {
                    required: true,
                  },
                ]}
              >
                <InputNumber
                  type="Diastolic_BP"
                  style={{border: "1px solid gold"}}
                />
    </Form.Item>
        </Col>
 
 
 
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Cholestrol"
                name="Cholestrol"
                rules={[
                  {
                    required: true,
                  },
                ]}
              >
                <InputNumber
                  type="Cholestrol"
                  style={{border: "1px solid gold"}}
                />
         </Form.Item>
        </Col>
 
 
 
        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Glucose"
                name="Glucose"
                rules={[
                  {
                    required: true,
                  },
                ]}
              >
                <InputNumber
                  type="Glucose"
                  style={{border: "1px solid gold"}}
                  />
    </Form.Item>
        </Col>

        <Col className="gutter-box" xs={24} sm={12} md={8} lg={6} xl={6}>
          <Form.Item
                label="Alcohol"
                name="Alcohol"
                rules={[
                  {
                    required: true,
                  },
                ]}
              >
          <Radio.Group>
            <Radio value="Yes" className="hover-color"> Yes </Radio><p>Yes</p>
            <Radio value="No" className="hover-color"> No </Radio>
         </Radio.Group>
               
         </Form.Item>
        </Col>
    </Row>  
   
    <Form.Item className="FormButtonClass">
                <Button type="primary" className="reset" danger onClick={() => {FormInsert.resetFields();setSpeech("")}}>
                  Reset
                </Button>
                <Button type="primary" className="submit" htmlType="submit" style={{ marginLeft: "10px" }}>
                  Submit
                </Button>
            </Form.Item>
 
    </Form>
    <div style={{ 
      display: 'block', width: 700, padding: 30 
    }}> 
      <> 
        <Modal title="Modal Title"
          visible={isModalVisible} 
          onOk={() => { 
            setIsModalVisible(false); 
          }} 
          onCancel={() => { 
            setIsModalVisible(false); 
          }}> 
          <p>{speech}</p> 
  
        </Modal> 
      </> 
    </div> 
    <ToastContainer/>
    </div>
  );
}
 
export default App;