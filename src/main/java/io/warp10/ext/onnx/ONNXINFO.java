//
//   Copyright 2023  SenX S.A.S.
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//

package io.warp10.ext.onnx;

import ai.onnxruntime.MapInfo;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxModelMetadata;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.SequenceInfo;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;
import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

public class ONNXINFO extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  public ONNXINFO(String name) {
    super(name);
  }

  public static final String METADATA = "model.metadata";
  public static final String PRODUCER_NAME = "producer.name";
  public static final String GRAPH_NAME = "graph.name";
  public static final String GRAPH_DESCRIPTION = "graph.description";
  public static final String DOMAIN = "domain";
  public static final String DESCRIPTION = "description";
  public static final String VERSION = "version";
  public static final String CUSTOM_METADATA = "custom.metadata";
  public static final String INPUT = "input.info";
  public static final String OUTPUT = "output.info";
  public static final String NODE_NAME = "name";
  public static final String NODE_TYPE = "info.class";
  public static final String NODE_INFO = "info";
  public static final String TENSOR_JAVA_TYPE = "java.type";
  public static final String TENSOR_ONNX_TYPE = "onnx.type";
  public static final String TENSOR_SHAPE = "shape";
  public static final String SEQUENCE_LENGTH = "length";
  public static final String SEQUENCE_TYPE = "type";
  public static final String MAP_SIZE = "size";
  public static final String MAP_KEY_TYPE = "key.type";
  public static final String MAP_VALUE_TYPE = "value.type";

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();

    if (!(top instanceof OrtSession)) {
      throw new WarpScriptException(getName() + " operates on an ONNX session.");
    }
    
    OrtSession session = (OrtSession) top;
    Map<String, Object> info = new LinkedHashMap<String, Object>(3);
    try {
      OnnxModelMetadata metadata = session.getMetadata();
      Map<String, Object> modelMetadata = new LinkedHashMap<String, Object>(7);
      modelMetadata.put(PRODUCER_NAME, metadata.getProducerName());
      modelMetadata.put(GRAPH_NAME, metadata.getGraphName());
      modelMetadata.put(GRAPH_DESCRIPTION, metadata.getDescription());
      modelMetadata.put(DOMAIN, metadata.getDomain());
      modelMetadata.put(DESCRIPTION, metadata.getDescription());
      modelMetadata.put(VERSION, metadata.getVersion());
      modelMetadata.put(CUSTOM_METADATA, metadata.getCustomMetadata());
      info.put(METADATA, modelMetadata);

      Map<String, NodeInfo> inputInfo = session.getInputInfo();
      Map<String, Object> inputResult = new LinkedHashMap<String, Object>((int) session.getNumInputs());
      for (Entry<String, NodeInfo> entry: inputInfo.entrySet()) {
        NodeInfo nodeInfo = entry.getValue();
        Map<String, Object> nodeResult = new LinkedHashMap<String, Object>(3);
        nodeResult.put(NODE_NAME, nodeInfo.getName());
        nodeResult.put(NODE_TYPE, nodeInfo.getInfo().getClass().getSimpleName());
        nodeResult.put(NODE_INFO, valueInfoToMap(nodeInfo.getInfo()));
        inputResult.put(entry.getKey(), nodeResult);
      }
      info.put(INPUT, inputResult);

      Map<String, NodeInfo> outputInfo = session.getOutputInfo();
      Map<String, Object> outputResult = new LinkedHashMap<String, Object>((int) session.getNumOutputs());
      for (Entry<String, NodeInfo> entry: outputInfo.entrySet()) {
        NodeInfo nodeInfo = entry.getValue();
        Map<String, Object> nodeResult = new LinkedHashMap<String, Object>(3);
        nodeResult.put(NODE_NAME, nodeInfo.getName());
        nodeResult.put(NODE_TYPE, nodeInfo.getInfo().getClass().getSimpleName());
        nodeResult.put(NODE_INFO, valueInfoToMap(nodeInfo.getInfo()));
        outputResult.put(entry.getKey(), nodeResult);
      }
      info.put(OUTPUT, outputResult);

    } catch (Exception oe) {
      throw new WarpScriptException(getName() + " encountered an error while reading model information.", oe);
    }

    stack.push(info);

    return stack;
  }

  private Map<String, Object> valueInfoToMap(ValueInfo valueInfo) {
    Map<String, Object> result = new LinkedHashMap<>();

    if (valueInfo instanceof TensorInfo) {
      TensorInfo tensorInfo = (TensorInfo) valueInfo;
      result.put(TENSOR_JAVA_TYPE, tensorInfo.type.toString());
      result.put(TENSOR_ONNX_TYPE, tensorInfo.onnxType.toString());

      long[] shape = tensorInfo.getShape();
      List<Long> shapeList = new ArrayList<Long>(shape.length);
      for (int i = 0; i < shape.length; i++) {
        shapeList.add(shape[i]);
      }
      result.put(TENSOR_SHAPE, shapeList);

    } else if (valueInfo instanceof SequenceInfo) {
      SequenceInfo sequenceInfo = (SequenceInfo) valueInfo;
      result.put(SEQUENCE_LENGTH, new Long(sequenceInfo.length));
      if (sequenceInfo.isSequenceOfMaps()) {
        result.put(SEQUENCE_TYPE, valueInfoToMap(sequenceInfo.mapInfo));
      } else {
        result.put(SEQUENCE_TYPE, sequenceInfo.sequenceType.toString());
      }

    } else if (valueInfo instanceof MapInfo) {
      MapInfo mapInfo = (MapInfo) valueInfo;
      result.put(MAP_SIZE, mapInfo.size);
      result.put(MAP_KEY_TYPE, mapInfo.keyType);
      result.put(MAP_VALUE_TYPE, mapInfo.valueType);

    } else {
      result.put("unknown.valueInfo.subclass", valueInfo.toString());
    }

    return result;
  }
  
}