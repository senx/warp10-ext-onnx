package io.warp10.ext.onnx;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

import org.apache.commons.io.IOUtils;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.warp10.script.NamedWarpScriptFunction;
import io.warp10.script.WarpScriptException;
import io.warp10.script.WarpScriptStack;
import io.warp10.script.WarpScriptStack.Macro;
import io.warp10.script.WarpScriptStackFunction;

public class ONNX extends NamedWarpScriptFunction implements WarpScriptStackFunction {
  
  private static final String ONNX_MODEL = "model";
  
  public ONNX(String name) {
    super(name);
  }
  
  @Override
  public Object apply(WarpScriptStack stack) throws WarpScriptException {
    
    Object top = stack.pop();
    
    if (!(top instanceof Map)) {
      throw new WarpScriptException(getName() + " expects a parameter MAP.");
    }
    
    Map<Object,Object> params = (Map<Object,Object>) top;
    
    top = stack.pop();
    
    if (!(top instanceof Macro)) {
      throw new WarpScriptException(getName() + " operates on a MACRO.");
    }
    
    Macro macro = (Macro) top;
    
    OrtEnvironment env = null;
    
    OrtSession session = null;
    
    try {
      env = OrtEnvironment.getEnvironment();
      env.setTelemetry(false);
      
      Object model = params.get(ONNX_MODEL);

      if (model instanceof byte[]) {
        session = env.createSession((byte[]) model);
      } else if (model instanceof String) {
        
        if (null == ONNXWarpScriptExtension.getModelRoot() && !ONNXWarpScriptExtension.isClassPathEnabled()) {
          throw new WarpScriptException(getName() + " model loading from directory or classpath not enabled.");
        }
        
        String path = (String) model;
        
        if (!(path.endsWith(".onnx"))) {
          throw new WarpScriptException(getName() + " model path does not end in '.onnx'.");
        }

        if (path.contains("./") || path.startsWith("/")) {
          throw new WarpScriptException(getName() + " invalid model path '" + path + "'.");
        }

        String root = ONNXWarpScriptExtension.getModelRoot();
        
        if (null != root) {
          File f = new File(root + "/" + path);
          
          if (f.exists()) {
            session = env.createSession(root + "/" + path);            
          }
        }
        
        // Check in the classpath
        if (null == session && ONNXWarpScriptExtension.isClassPathEnabled()) {
          InputStream in = this.getClass().getResourceAsStream(path);
          
          if (null != in) {
            byte[] buf = new byte[1024];
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            while(true) {
              int len = in.read(buf);
              if (len < 0) {
                break;
              }
              out.write(buf, 0, len);
            }
            in.close();
            
            session = env.createSession(out.toByteArray());
          }
        }
        
        if (null == session) {
          throw new WarpScriptException(getName() + " failed to load ONNX model '" + path + "'.");
        }
      } else {
        throw new WarpScriptException(getName() + " invalid '" + ONNX_MODEL + "' entry, expected BYTES or STRING.");
      }
            
      stack.push(session);
      stack.exec(macro);
    } catch (IOException ioe) {
      throw new WarpScriptException(getName() + " error loading ONNX model.", ioe);
    } catch (OrtException oe) {
      throw new WarpScriptException(getName() + " error loading ONNX model.", oe);
    } finally {
      WarpScriptException error = null;
      
      if (null != session) {       
        try {
          session.close();
        } catch (OrtException oe) {
          error = new WarpScriptException(getName() + " error while closing ONNC session,", oe);
        }
      }
      
      if (null != env) {
        try {
          env.close();
        } catch (OrtException oe) {
          if (null == error) {
            error = new WarpScriptException(getName() + " error while closing ONNX environment.", oe);
          }
        }
      }
      
      if (null != error) {
        throw error;
      }
    }
    
    return stack;
  }

}
