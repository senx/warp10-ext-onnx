package io.warp10.ext.onnx;

import java.util.List;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStackFunction;

public class TOONNX extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  public TOONNX(String name) {
    super(name);
  }

  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    Object top = stack.pop();
    
    try {
      if (top instanceof List) {
        List<Object> list = (List<Object>) top;
        
        Object data = ONNXUtils.toArray(list);
            
        OnnxTensor tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), data);
        stack.push(tensor);
      } else if (top instanceof Long || top instanceof Double || top instanceof String) {
        OnnxTensor tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), top);
        stack.push(tensor);
      } else {
        throw new WarpScriptException(getName() + " operates on a LIST, STRING, LONG or DOUBLE.");
      }      
    } catch (OrtException oe) {
      throw new WarpScriptException("Error while converting to an ONNX Tensor.", oe);
    }
        
    return stack;
  }
  
}
