import json
import subprocess
import cv2
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render,HttpResponse

# from rest_framework.response import Response

# Create your views here.
def change_hostname(new_hostname):
    # Update /etc/hostname
    with open('/etc/hostname', 'w') as f:
        f.write(new_hostname + '\n')

    # Update /etc/hosts
    with open('/etc/hosts', 'r') as f:
        hosts_content = f.readlines()

    with open('/etc/hosts', 'w') as f:
        for line in hosts_content:
            if '127.0.1.1' in line:
                line = f'127.0.1.1\t{new_hostname}\n'
            f.write(line)

    # Change the hostname immediately
    subprocess.run(['hostnamectl', 'set-hostname', new_hostname])

def set_static_ip(connection_name, static_ip, gateway_ip, dns_ip):
    # Set static IP
    subprocess.run(['nmcli', 'con', 'modify', connection_name, 'ipv4.addresses', static_ip])
    subprocess.run(['nmcli', 'con', 'modify', connection_name, 'ipv4.gateway', gateway_ip])
    subprocess.run(['nmcli', 'con', 'modify', connection_name, 'ipv4.dns', dns_ip])
    subprocess.run(['nmcli', 'con', 'modify', connection_name, 'ipv4.method', 'manual'])

    # Restart the connection
    subprocess.run(['nmcli', 'con', 'down', connection_name])
    subprocess.run(['nmcli', 'con', 'up', connection_name])

    print(f'Static IP set to {static_ip} for {connection_name}.')




def set_dynamic_ip(connection_name):
    # Set to DHCP
    subprocess.run(['nmcli', 'con', 'modify', connection_name, 'ipv4.method', 'auto'])

    # Restart the connection
    subprocess.run(['nmcli', 'con', 'down', connection_name])
    subprocess.run(['nmcli', 'con', 'up', connection_name])

    print(f'Dynamic IP set for {connection_name}.')



def video_stream():
    # picam2 = Picamera2()
    # picam2.start()
    # Open the camera
    camera = cv2.VideoCapture(0)  # Change to the appropriate camera index if needed
    while True:
        # Read frame from the camera
        # frame = picam2.capture_array()
        ret, frame = camera.read()
        if not ret:
                break

        # Encode the frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        # Yield the frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def stream(request):
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')


def update_threshold(request):
    data = request.data
    print(data)
    print('new thresh',data.threshold)
    with open('config.json','rb') as file:
        data = json.load(file)
        print(data)
    return HttpResponse(data['threshold'])


def get_threshold(request):
    with open('config.json','rb') as file:
        data = json.load(file)
        print(data)
    return HttpResponse(data['threshold'])


def get_slot_details(request):
    return JsonResponse({
        'NoOfSlots':3
    })

