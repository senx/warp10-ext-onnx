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
    functions.put("ONNX.TENSOR", new ONNXTENSOR("ONNX.TENSOR"));
    functions.put("ONNX.INFO", new ONNXINFO("ONNX.INFO"));
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
