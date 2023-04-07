package io.warp10.ext.onnx;

import java.util.HashMap;
import java.util.Map;

import io.warp10.WarpConfig;
import io.warp10.warp.sdk.WarpScriptExtension;

public class ONNXWarpScriptExtension extends WarpScriptExtension {
  
  public static final String CONF_MODEL_ROOT = "onnx.modelroot";
  public static final String CONF_CLASSPATH = "onnx.classpath";
  
  private static final Map<String,Object> functions;
  
  private static final String modelRoot;
  
  private static final boolean classPathEnabled;
  
  static {
    modelRoot = WarpConfig.getProperty(CONF_MODEL_ROOT, null);
    classPathEnabled = "true".equals(WarpConfig.getProperty(CONF_CLASSPATH));

    functions = new HashMap<String,Object>();
    
    functions.put("ONNX", new ONNX("ONNX"));
    functions.put("ONNX.RUN", new ONNXRUN("ONNX.RUN"));
    functions.put("->ONNX", new TOONNX("->ONNX"));    
  }
  
  @Override
  public Map<String, Object> getFunctions() {
    return functions;
  }
  
  public static String getModelRoot() {
    return modelRoot;
  }
  
  public static boolean isClassPathEnabled() {
    return classPathEnabled;
  }
}
