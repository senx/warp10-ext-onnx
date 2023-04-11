//
//   Copyright 2020-2023  SenX S.A.S.
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

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ai.onnxruntime.OnnxMap;
import ai.onnxruntime.OnnxSequence;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import io.warp10.script.WarpScriptException;

public class ONNXUtils {

  public static Object fromONNXValue(OnnxValue value) throws WarpScriptException {
    try {
      switch(value.getType()) {
        case ONNX_TYPE_MAP:
          //Map<Object,Object> map = ((OnnxMap) value).getValue();
          Map map = ((OnnxMap) value).getValue();
          return sanitize(map);
        case ONNX_TYPE_SEQUENCE:
          //List<Object> list = ((OnnxSequence) value).getValue();
          List list = ((OnnxSequence) value).getValue();
          return sanitize(list);
        case ONNX_TYPE_TENSOR:
        case ONNX_TYPE_SPARSETENSOR:
          Object tensor = ((OnnxTensor) value).getValue();
          return sanitize(tensor);
        case ONNX_TYPE_OPAQUE:
        case ONNX_TYPE_UNKNOWN:
        default:
          throw new WarpScriptException("Unsupported ONNX Value type " + value.getType().name());
      }
    } catch (OrtException oe) {
      throw new WarpScriptException("Error while converting ONNX value.", oe);
    }
  }

  private static Object sanitize(Object input) throws WarpScriptException {
    if (input instanceof Map) {
      // Check if any of the values is of type Float, if so we need to re-allocate a map
      boolean reallocate = false;
      for (Object o: ((Map) input).values()) {
        if (o instanceof Float) {
          reallocate = true;
          break;
        }
      }
      if (!reallocate) {
        return input;
      }

      Map<Object,Object> newmap = new HashMap<Object,Object>(((Map) input).size());
      for (Entry<Object,Object> entry: ((Map<Object,Object>) input).entrySet()) {
        if (entry.getValue() instanceof Float) {
          newmap.put(entry.getKey(), ((Float) entry.getValue()).doubleValue());
        } else {
          newmap.put(entry.getKey(), entry.getValue());
        }
      }
      return newmap;
    } else if (input instanceof List) {
      List<Object> newlist = new ArrayList<Object>(((List) input).size());
      for (Object elt: (List) input) {
        newlist.add(sanitize(elt));
      }
      return newlist;
    } else if (input instanceof Long || input instanceof Double || input instanceof String || input instanceof Boolean) {
      return input;
    } else if (input instanceof Integer || input instanceof Short || input instanceof Byte) {
      return ((Number) input).longValue();
    } else if (input instanceof Float) {
      return ((Float) input).doubleValue();
    } else if (input.getClass().isArray()) {
      int size = Array.getLength(input);
      List<Object> array = new ArrayList<Object>(size);
      for (int i = 0; i < size; i++) {
        array.add(sanitize(Array.get(input, i)));
      }
      return array;
    } else {
      throw new WarpScriptException("Unsupported type " + input.getClass());
    }
  }

  public static Object toArray(List<Object> list) throws WarpScriptException {
    if (list.isEmpty()) {
      return new Object[0];
    }

    // Determine type of elements if possible
    Object first = list.get(0);

    if (first instanceof Long) {
      long[] array = new long[list.size()];
      for (int i = 0; i < list.size(); i++) {
        array[i] = ((Long) list.get(i)).longValue();
      }
      return array;
    } else if (first instanceof Double) {
      double[] array = new double[list.size()];
      for (int i = 0; i < list.size(); i++) {
        array[i] = ((Double) list.get(i)).doubleValue();
      }
      return array;
    } else if (first instanceof String) {
      String[] array = new String[list.size()];
      for (int i = 0; i < list.size(); i++) {
        array[i] = (String) list.get(i);
      }
      return array;
    } else if (first instanceof List) {
      Object[] array = new Object[list.size()];
      for (int i = 0; i < list.size(); i++) {
        array[i] = toArray((List<Object>) list.get(i));
      }
      return array;
    } else {
      throw new WarpScriptException("Invalid type, cannot convert " + first.getClass() + " to an ONNX Tensor.");
    }
  }
}
