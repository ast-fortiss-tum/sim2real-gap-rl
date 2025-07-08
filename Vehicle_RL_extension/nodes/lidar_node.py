#!/usr/bin/env python3

import requests
import asyncio
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaStreamTrack, MediaRecorder

class SignalingClient:
    def __init__(self, server_url):
        self.server_url = server_url

    def retrieve_offer(self):
        try:
            response = requests.get(self.server_url + '/getOffer')
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving offer: {e}")
            return None

    def send_answer(self, answer):
        try:
            response = requests.post(
                self.server_url + '/answer',
                json=answer,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            print("Answer sent to server.")
        except requests.exceptions.RequestException as e:
            print(f"Error sending answer: {e}")

def get_metadata(server_url):
    try:
        response = requests.get(server_url + '/metadata')
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Could not retrieve the intrinsic matrix: {e}")
        return None

async def start_receiving_stream(server_url):
    recorder = MediaRecorder("video.mp4")
    signaling_client = SignalingClient(server_url)
    peer_connection = RTCPeerConnection()

    # @peer_connection.on("icecandidate")
    # async def on_icecandidate(candidate):
    #     print("Peer called ice candidate")
    #     if candidate is None:
    #         answer = {
    #             'type': 'answer',
    #             'data': peer_connection.localDescription.sdp
    #         }
    #         signaling_client.send_answer(answer)

    def add_tracks():
        peer_connection.addTrack(FlagVideoStreamTrack())

    @peer_connection.on("track")
    def on_track(track):
        # print("Receiving video track")
        # print(track)
        # if track.kind == "video":
        #     asyncio.ensure_future(display_video(track))
        print("Receiving %s" % track.kind)
        recorder.addTrack(track)

    # Retrieve the offer from the remote server
    remote_offer = signaling_client.retrieve_offer()
    if remote_offer is None:
        print("Failed to retrieve offer.")
        return
    print("Acquired offer from iDevice")
    #print(remote_offer)
    
    # Set the remote description
    await peer_connection.setRemoteDescription(
        RTCSessionDescription(sdp=remote_offer["sdp"], type=remote_offer["type"])
    )
    print("Remote description set")

    # Create and set the local description
    await peer_connection.setLocalDescription(await peer_connection.createAnswer())

    print("Waiting for ICE candidate gathering to finish...")

    answer = {
        'type': 'answer',
        'data': peer_connection.localDescription.sdp
    }
    signaling_client.send_answer(answer)

    while True:
        pass



async def display_video(track):
    print("entered display")
    while True:
        frame = await track.recv()
        print(frame)

        # Convert the frame to a numpy array
        img = frame.to_ndarray(format="bgr24")

        # Display the image using OpenCV
        cv2.imshow("Received Video", img)

        cv2.waitKey(0)

        # Exit on pressing 'q'
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cv2.destroyAllWindows()

# Main entry point
async def main():
    remote_address = "http://192.168.0.100"

    # Get metadata (optional)
    #get_metadata(remote_address)

    # Start receiving the stream
    await start_receiving_stream(remote_address)

if __name__ == "__main__":
    asyncio.run(main())
