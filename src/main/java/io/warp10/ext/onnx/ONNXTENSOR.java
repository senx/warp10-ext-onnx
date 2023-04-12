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

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.ShortBuffer;
import java.util.List;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

public class ONNXTENSOR extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  public ONNXTENSOR(String name) {
    super(name);
  }

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    Object top = stack.pop();
    if (!(top instanceof List)) {
      throw new WarpScriptException(getName() + " expects a shape (LIST)");
    }

    List list = (List) top;
    long[] shape = new long[list.size()];
    for (int i = 0; i < list.size(); i++) {
      if (!((list.get(i)) instanceof Long)) {
        throw new WarpScriptException("Shape argument has a non LONG object at pos " + i);
      }
      shape[i] = (Long) list.get(i);
    }

    top = stack.pop();
    if (!(top instanceof String)) {

      StringBuilder sb = new StringBuilder();
      for (OnnxJavaType ot: OnnxJavaType.values()) {
        sb.append(" ");
        sb.append(ot.toString());
      }
      sb.append(" ]");
      throw new WarpScriptException(getName() + " expects an ONNX JAVA TYPE as a STRING. One of: [" + sb);
    }
    OnnxJavaType type = OnnxJavaType.valueOf((String) top);

    top = stack.pop();
    if (!(top instanceof List || top instanceof byte[])) {
      throw new WarpScriptException(getName() + " expects input data as LIST or BYTES");
    }

    OnnxTensor tensor;
    try {

      if (top instanceof byte[]) {
        tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), ByteBuffer.wrap((byte[]) top), shape, type);

      } else {
        List data = (List) top;
        int size = flattenSize(data);

        switch (type) {
          case FLOAT:
            FloatBuffer floatBuffer = FloatBuffer.allocate(size);
            loadFLOAT(data, floatBuffer);
            floatBuffer.rewind();
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), floatBuffer, shape);
            break;
          case DOUBLE:
            DoubleBuffer doubleBuffer = DoubleBuffer.allocate(size);
            loadDOUBLE(data, doubleBuffer);
            doubleBuffer.rewind();
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), doubleBuffer, shape);
            break;
          case UINT8:
          case INT8:
            ByteBuffer byteBuffer = ByteBuffer.allocate(size);
            loadINT8(data, byteBuffer);
            byteBuffer.rewind();
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), byteBuffer, shape, type);
            break;
          case INT16:
            ShortBuffer shortBuffer = ShortBuffer.allocate(size);
            loadINT16(data, shortBuffer);
            shortBuffer.rewind();
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), shortBuffer, shape);
            break;
          case INT32:
            IntBuffer intBuffer = IntBuffer.allocate(size);
            loadINT32(data, intBuffer);
            intBuffer.rewind();
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), intBuffer, shape);
            break;
          case INT64:
            LongBuffer longBuffer = LongBuffer.allocate(size);
            loadINT64(data, longBuffer);
            longBuffer.rewind();
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), longBuffer, shape);
            break;
          case BOOL:
            ByteBuffer boolBuffer = ByteBuffer.allocate(size);
            loadBOOL(data, boolBuffer);
            boolBuffer.rewind();
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), boolBuffer, shape, type);
            break;
          case STRING:
            String[] a = new String[size];
            loadSTRING(data, a, 0);
            tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), a, shape);
            break;
          case UNKNOWN:
          default:
            throw new WarpScriptException(getName() + " got an unknown ONNX JAVA TYPE");
        }

      }

    } catch (OrtException oe) {
      throw new WarpScriptException("Error while creating an ONNX Tensor.", oe);
    }

    stack.push(tensor);

    return stack;
  }

  private int flattenSize(List list) {
    int c = 0;
    for (Object o: list) {
      if (o instanceof List) {
        c += flattenSize((List) o);
      } else {
        c++;
      }
    }

    return c;
  }

  private void loadFLOAT(List l, FloatBuffer b){
    for(Object o: l) {
      if (o instanceof List) {
        loadFLOAT((List) o, b);
      } else {
        float x = ((Number) o).floatValue();
        b.put(x);
      }
    }
  }

  private void loadDOUBLE(List l, DoubleBuffer b){
    for(Object o: l) {
      if (o instanceof List) {
        loadDOUBLE((List) o, b);
      } else {
        double x = ((Number) o).doubleValue();
        b.put(x);
      }
    }
  }

  private void loadINT8(List l, ByteBuffer b){
    for(Object o: l) {
      if (o instanceof List) {
        loadINT8((List) o, b);
      } else {
        byte x = ((Long) o).byteValue();
        b.put(x);
      }
    }
  }

  private void loadINT16(List l, ShortBuffer b){
    for(Object o: l) {
      if (o instanceof List) {
        loadINT16((List) o, b);
      } else {
        short x = ((Long) o).shortValue();
        b.put(x);
      }
    }
  }

  private void loadINT32(List l, IntBuffer b){
    for(Object o: l) {
      if (o instanceof List) {
        loadINT32((List) o, b);
      } else {
        int x = ((Long) o).intValue();
        b.put(x);
      }
    }
  }

  private void loadINT64(List l, LongBuffer b){
    for(Object o: l) {
      if (o instanceof List) {
        loadINT64((List) o, b);
      } else {
        b.put((Long) o);
      }
    }
  }

  private void loadBOOL(List l, ByteBuffer b){
    // we assume here 1 boolean per byte
    for(Object o: l) {
      if (o instanceof List) {
        loadBOOL((List) o, b);
      } else {
        b.put((byte) ((Boolean) o ? 1 : 0));
      }
    }
  }

  private int loadSTRING(List l, String[] a, int pos){
    for(Object o: l) {
      if (o instanceof List) {
        pos = loadSTRING((List) o, a, pos);
      } else {
        a[pos++] = (String) o;
      }
    }

    return pos;
  }
  
}
