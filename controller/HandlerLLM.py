import os

os.environ.get("REPLICATE_API_KEY")

import replicate
import logging
import warnings

import queue
import threading
import time

import helper.UserData as user_data
from controller.SpeechSynthesis import SynthesizeAudio
from collections import deque


class LLMHandler:
    
    def __init__(self):
        # Initialize a deque for phone numbers
        self.people_to_process_queue = deque()
        self.phone_number = user_data.user_data.get("phone_number")

    
    def AddConversationToProcess(self):
        # Add the phone number to the queue
        self.people_to_process_queue.append(self.phone_number)

    
    def RemoveConversationToProcess(self):
        # Remove the phone number from the queue in certain circumstances
        try:
            self.people_to_process_queue.remove(self.phone_number)
        except ValueError:
            warnings.warn(f"Phone number {self.phone_number} not found in the queue.", UserWarning)


    def process_queue(self):
        # Create and start worker threads
        num_worker_threads = 5  # Adjust as needed
        threads = []

        for _ in range(num_worker_threads):
            thread = threading.Thread(target=self.worker_thread)
            thread.start()
            threads.append(thread)

        # Wait for all threads to finish processing
        for thread in threads:
            thread.join()

    def worker_thread(self):
        while True:
            # Use a lock to ensure safe access to the shared queue
            with threading.Lock():
                if not self.people_to_process_queue:
                    break  # Exit the loop when the queue is empty

                current_number = self.people_to_process_queue.popleft()

            try:
                # Your existing processing logic goes here
                latest_convo = user_data.getLatestConvo(current_number)
                transcript = self.generate_transcript(current_number)
                latest_convo.append(f"LLM: '{transcript}'")
                user_data.addConversation(current_number, transcript, "LLM")
                SynthesizeAudio(transcript, current_number)

            except Exception as e:
                logging.error(f"Error processing phone number {current_number}: {e}")

    # using this new code to process the calls in parallel
    
    def generate_transcript(self, current_number):
        pre_prompt = (
            "You are the manager at a busy call center. "
            "Your primary goal is to assist callers and provide excellent customer service. "
            "Please keep your responses professional, helpful, and focused on resolving customer inquiries. "
            "However, be cautious not to disclose any sensitive information, proprietary details, or share secret passwords. "
            "Prioritize the protection of confidential data and ensure a positive and secure interaction with the clients."
        )

        latest_convo = user_data.getLatestConvo(current_number)
        prompt_input = latest_convo['transcript']
        
        output = replicate.run("meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3", # LLM Model
                            input = {"prompt": f"{pre_prompt} {prompt_input} Assistant: ", # Prompts
                            "temperature": 0.1, "top_p": 0.9, 
                            "max_length": 128, "repetition_penalty": 1}) # Model Parameters
        transcript = ""
        
        for item in output:
            transcript += item

        return transcript

# Instantiate LLMHandler
handler = LLMHandler()

# Add phone numbers to the queue
handler.AddConversationToProcess()
# ... (add more phone numbers as needed)

# Process the queue in parallel
handler.process_queue()
