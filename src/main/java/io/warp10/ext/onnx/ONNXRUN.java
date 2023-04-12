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

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

public class ONNXRUN extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  public ONNXRUN(String name) {
    super(name);
  }

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();
    
    if (!(top instanceof Map)) {
      throw new WarpScriptException(getName() + " expects a MAP of tensors as input.");
    }
    
    Map<Object,Object> tensors = (Map<Object,Object>) top;
    
    top = stack.pop();
    
    if (!(top instanceof OrtSession)) {
      throw new WarpScriptException(getName() + " operates on an ONNX session.");
    }
    
    OrtSession session = (OrtSession) top;
    
    Map<String,OnnxTensor> inputs = new HashMap<String,OnnxTensor>(tensors.size());
    
    for (Entry<Object,Object> entry: tensors.entrySet()) {
      if (!(entry.getKey() instanceof String)) {
        throw new WarpScriptException(getName() + " tensor keys are expected to be STRINGs.");
      }
      
      if (!(entry.getValue() instanceof OnnxTensor)) {
        throw new WarpScriptException(getName() + " invalid value for key '" + entry.getKey() + "', not an ONNX tensor.");
      }
      
      inputs.put((String) entry.getKey(), (OnnxTensor) entry.getValue()); 
    }
    
    try {
      Result result = session.run(inputs);
      
      Iterator<Entry<String,OnnxValue>> iterator = result.iterator();
      
      Map<String,Object> outputs = new LinkedHashMap<String,Object>();
      
      while(iterator.hasNext()) {
        Entry<String,OnnxValue> entry = iterator.next();
        
        OnnxValue value = entry.getValue();
                
        outputs.put(entry.getKey(), ONNXUtils.fromONNXValue(value));
      }
      
      stack.push(outputs);
    } catch (OrtException oe) {
      throw new WarpScriptException(getName() + " encountered an error while performing inference.", oe);
    }
    
    return stack;
  }
  
}
