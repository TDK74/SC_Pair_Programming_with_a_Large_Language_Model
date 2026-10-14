import os
import google.generativeai as genai

from google.api_core import client_options as client_options_lib
from utils import get_api_key


## ------------------------------------------------------ ##
genai.configure(
                api_key = get_api_key(),
                transport = "rest",
                client_options = client_options_lib.ClientOptions(
                    api_endpoint = os.getenv("GOOGLE_API_BASE"),
                    )
                )

## ------------------------------------------------------ ##
models = [m for m in genai.list_models()
          if 'generateText' in m.supported_generation_methods]
model_bison = models[0]
model_bison

## ------------------------------------------------------ ##
model_flash = genai.GenerativeModel(model_name = 'gemini-1.5-flash')

## ------------------------------------------------------ ##
def generate_text(prompt, model = model_flash, temperature = 0.0):
    return model_flash.generate_content(prompt,
                                        generation_config =
                                        {'temperature':temperature})

## ------------------------------------------------------ ##
# Note: Taken from https://github.com/lmoroney/odmlbook/blob/
# 63c0825094b2f44efc5c4d3226425a51990e73d6/BookSource/Chapter08/ios/cats_vs_dogs/
# CatVsDogClassifierSample/ModelDataHandler/ModelDataHandler.swift
CODE_BLOCK = """
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import CoreImage
import TensorFlowLite
import UIKit


/// An inference from invoking the `Interpreter`.
struct Inference {
  let confidence: Float
  let label: String
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet model.
enum MobileNet {
  static let modelInfo: FileInfo = (name: "converted_model", extension: "tflite")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler {

  // MARK: - Public Properties

  /// The current thread count used by the TensorFlow Lite Interpreter.
  let threadCount: Int

  let resultCount = 1

  // MARK: - Model Parameters

  let batchSize = 1
  let inputChannels = 3
  let inputWidth = 224
  let inputHeight = 224

  // MARK: - Private Properties

  /// List of labels from the given labels file.
  private var labels: [String] = ["Cat", "Dog"]

  /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
  private var interpreter: Interpreter

  /// Information about the alpha component in RGBA data.
  private let alphaComponent = (baseOffset: 4, moduloRemainder: 3)

  // MARK: - Initialization

  /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
  /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
  init?(modelFileInfo: FileInfo, threadCount: Int = 1) {
    let modelFilename = modelFileInfo.name

    // Construct the path to the model file.
    guard let modelPath = Bundle.main.path(
      forResource: modelFilename,
      ofType: modelFileInfo.extension
      ) else {
        print("Failed to load the model file with name: \(modelFilename).")
        return nil
    }

    // Specify the options for the `Interpreter`.
    self.threadCount = threadCount
    var options = InterpreterOptions()
    options.threadCount = threadCount
    do {
      // Create the `Interpreter`.
      interpreter = try Interpreter(modelPath: modelPath, options: options)
    } catch let error {
      print("Failed to create the interpreter with error: \(error.localizedDescription)")
      return nil
    }

  }

  // MARK: - Public Methods

  /// Performs image preprocessing, invokes the `Interpreter`, and process the inference results.
  func runModel(onFrame pixelBuffer: CVPixelBuffer) -> [Inference]? {
    let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
    assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
      sourcePixelFormat == kCVPixelFormatType_32BGRA ||
      sourcePixelFormat == kCVPixelFormatType_32RGBA)


    let imageChannels = 4
    assert(imageChannels >= inputChannels)

    // Crops the image to the biggest square in the center and scales it down to model dimensions.
    let scaledSize = CGSize(width: inputWidth, height: inputHeight)
    guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize) else {
      return nil
    }

    let outputTensor: Tensor
    do {
      // Allocate memory for the model's input `Tensor`s.
      try interpreter.allocateTensors()

      // Remove the alpha component from the image buffer to get the RGB data.
      guard let rgbData = rgbDataFromBuffer(
        thumbnailPixelBuffer,
        byteCount: batchSize * inputWidth * inputHeight * inputChannels
        ) else {
          print("Failed to convert the image buffer to RGB data.")
          return nil
      }

      // Copy the RGB data to the input `Tensor`.
      try interpreter.copy(rgbData, toInputAt: 0)

      // Run inference by invoking the `Interpreter`.
      try interpreter.invoke()

      // Get the output `Tensor` to process the inference results.
      outputTensor = try interpreter.output(at: 0)
    } catch let error {
      print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
      return nil
    }

    let results = [Float32](unsafeData: outputTensor.data) ?? []

    // Process the results.
    let topNInferences = getTopN(results: results)

    // Return the inference time and inference results.
    return topNInferences
  }

  // MARK: - Private Methods

  /// Returns the top N inference results sorted in descending order.
  private func getTopN(results: [Float]) -> [Inference] {
    // Create a zipped array of tuples [(labelIndex: Int, confidence: Float)].
    let zippedResults = zip(labels.indices, results)

    // Sort the zipped results by confidence value in descending order.
    let sortedResults = zippedResults.sorted { $0.1 > $1.1 }.prefix(resultCount)

    // Return the `Inference` results.
    return sortedResults.map { result in Inference(confidence: result.1, label: labels[result.0]) }
  }

  /// Loads the labels from the labels file and stores them in the `labels` property.
  private func loadLabels(fileInfo: FileInfo) {
    let filename = fileInfo.name
    let fileExtension = fileInfo.extension
    guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
      fatalError("Labels file not found in bundle. Please add a labels file with name " +
        "\(filename).\(fileExtension) and try again.")
    }
    do {
      let contents = try String(contentsOf: fileURL, encoding: .utf8)
      labels = contents.components(separatedBy: .newlines)
    } catch {
      fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
        "valid labels file and try again.")
    }
  }

  /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
  ///
  /// - Parameters
  ///   - buffer: The pixel buffer to convert to RGB data.
  ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
  ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
  ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
  ///       floating point values).
  /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
  ///     converted.
  private func rgbDataFromBuffer(
    _ buffer: CVPixelBuffer,
    byteCount: Int
    ) -> Data? {
    CVPixelBufferLockBaseAddress(buffer, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
    guard let mutableRawPointer = CVPixelBufferGetBaseAddress(buffer) else {
      return nil
    }
    let count = CVPixelBufferGetDataSize(buffer)
    let bufferData = Data(bytesNoCopy: mutableRawPointer, count: count, deallocator: .none)
    var rgbBytes = [Float](repeating: 0, count: byteCount)
    var index = 0
    for component in bufferData.enumerated() {
      let offset = component.offset
      let isAlphaComponent = (offset % alphaComponent.baseOffset) == alphaComponent.moduloRemainder
      guard !isAlphaComponent else { continue }
      rgbBytes[index] = Float(component.element) / 255.0
      index += 1
    }

    return rgbBytes.withUnsafeBufferPointer(Data.init)

  }
}

// MARK: - Extensions

extension Data {
  /// Creates a new buffer by copying the buffer pointer of the given array.
  ///
  /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
  ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
  ///     data from the resulting buffer has undefined behavior.
  /// - Parameter array: An array with elements of type `T`.
  init<T>(copyingBufferOf array: [T]) {
    self = array.withUnsafeBufferPointer(Data.init)
  }
}

extension Array {
  /// Creates a new array from the bytes of the given unsafe data.
  ///
  /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
  ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
  ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
  /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
  ///     `MemoryLayout<Element>.stride`.
  /// - Parameter unsafeData: The data containing the bytes to turn into an array.
  init?(unsafeData: Data) {

    guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
    #if swift(>=5.0)
    self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    #else
    self = unsafeData.withUnsafeBytes {
      .init(UnsafeBufferPointer<Element>(
        start: $0,
        count: unsafeData.count / MemoryLayout<Element>.stride
      ))
    }
    #endif  // swift(>=5.0)
  }
}
"""

## ------------------------------------------------------ ##
CODE_BLOCK = """
# replace this with your own code

# some code from a course for explanation

import numpy as np
import matplotlib.pyplot as plt

# Simulate motion with constant velocity and noisy position measurements

# Time step
dt = 1.0

# State transition matrix (constant velocity model)
F = np.array([[1, dt],
              [0, 1]])

# Observation matrix (we only measure the position)
H = np.array([[1, 0]])

# Process noise covariance matrix (uncertainty in model prediction)
Q = np.array([[1, 0],
              [0, 1]])

# Measurement noise covariance matrix (uncertainty in measurement)
R = np.array([[10]])

# Initial state estimate (position and velocity)
x = np.array([[0],  # Initial position
              [1]])  # Initial velocity

# Initial estimate covariance matrix
P = np.array([[500, 0],
              [0, 500]])

# Number of iterations (time steps)
n = 50

# Simulated true positions and noisy measurements
true_positions = []
measurements = []
velocities = []
position = 0
velocity = 1

for i in range(n):
    # True position and velocity (without noise)
    position += velocity * dt
    true_positions.append(position)

    # Noisy measurement (position)
    noisy_measurement = position + np.random.normal(0, np.sqrt(R[0, 0]))
    measurements.append(noisy_measurement)

# Kalman filter process
estimates = []  # Store position estimates

for i in range(n):
    # Prediction Step
    x = np.dot(F, x)  # State prediction
    P = np.dot(np.dot(F, P), F.T) + Q  # Covariance prediction

    # Measurement update step (if we have a measurement)
    z = np.array([[measurements[i]]])  # Current measurement (noisy position)
    y = z - np.dot(H, x)  # Innovation (residual)
    S = np.dot(H, np.dot(P, H.T)) + R  # Innovation covariance
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S))  # Kalman gain

    x = x + np.dot(K, y)  # Updated state estimate
    P = np.dot(np.eye(2) - np.dot(K, H), P)  # Updated covariance estimate

    # Store the estimate
    estimates.append(x[0, 0])

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(true_positions, label = 'True Position')
plt.plot(measurements, 'ro', label = 'Noisy Measurements')
plt.plot(estimates, 'b-', label = 'Kalman Filter Estimate')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Kalman Filter for Position Estimation')
plt.show()

"""

prompt_template = """
Can you please explain how this code works?

{question}

Use a lot of detail and make it as clear as possible.
"""

completion = generate_text(
                prompt = prompt_template.format(question = CODE_BLOCK)
                )
print(completion.text)

## ------------------------------------------------------ ##
prompt_template = """
Please write technical documentation for this code and \n
make it easy for a non ML/DL developer to understand:

{question}

Output the results in markdown
"""

## ------------------------------------------------------ ##
completion = generate_text(
                prompt = prompt_template.format(question = CODE_BLOCK)
                )
print(completion.text)

## ------------------------------------------------------ ##
CODE_BLOCK = """
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedSelfAttention(nn.Module):

    def __init__(self, d_model=2,
                 row_dim=0,
                 col_dim=1):

        super().__init__()

        self.W_q = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_k = nn.Linear(in_features=d_model, out_features=d_model, bias=False)
        self.W_v = nn.Linear(in_features=d_model, out_features=d_model, bias=False)

        self.row_dim = row_dim
        self.col_dim = col_dim


    def forward(self, token_encodings, mask=None):

        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))

        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)

        if mask is not None:
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)

        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores

"""

prompt_template = """
Please write technical documentation for this code and \n
make it easy for a non Python developer to understand:

{question}

Output the results in markdown
"""

completion = generate_text(
                prompt = prompt_template.format(question = CODE_BLOCK)
                )
print(completion.text)
