import cv2

MODEL_mean = (78.4463377603, 87.7689143744, 114.895847746)
age_list = ['(1,4)', '(5,10)', '(11,20)', '(21,30)', '(30,35)', '(36,40)', '(41,45)', '(46,50)', '(51,60)', '(61,70)']
gender_list = ['male', 'female']

def filesget():
    age_net = cv2.dnn.readNetFromCaffe('data/deploy_age.prototxt', 'data/age_net.caffemodel')
    gender_net = cv2.dnn.readNetFromCaffe('data/deploy_gender.prototxt', 'data/gender_net.caffemodel')
    return age_net, gender_net

def readfcam(age_net, gender_net):
    font = cv2.FONT_HERSHEY_COMPLEX
    imagei = cv2.imread('human/795736-1561228636.jpg')
    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
    gray = cv2.cvtColor(imagei, cv2.COLOR_BGR2GRAY)
    facese = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    if len(facese) > 0:
        print("Found {} faces".format(len(facese)))

    for (x, y, w, h) in facese:
        cv2.rectangle(imagei, (x, y), (x+w, y+h), (255, 255, 0), 2)
        face_img = imagei[y:y+h, x:x+w].copy()  # Corrected region extraction
        
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_mean, swapRB=False)
        
        # Gender prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]
        print("Gender:", gender)
        
        # Age prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]
        print("Age:", age)
        
        gender_age = "{} {}".format(gender, age)
        cv2.putText(imagei, gender_age, (x, y - 10), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imshow('AI Project by Tibyan', imagei)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    age_net, gender_net = filesget()
    readfcam(age_net, gender_net)
