# Альтернативная функция загрузки
@st.cache_resource
def load_model_alternative():
    weights_path = "models/enhanced_epoch_28_ratio_1.23.pth"
    
    if not os.path.exists(weights_path):
        return None, None
    
    try:
        # Способ 1: Используем pickle напрямую
        import pickle
        
        with open(weights_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        model = StrongGenerator().to(device)
        
        if 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        st.success("✅ Модель загружена через pickle")
        return model, device
        
    except Exception as e:
        st.error(f"Ошибка альтернативной загрузки: {e}")
        
        # Способ 2: Специальный контекстный менеджер
        try:
            import torch.serialization
            
            with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                checkpoint = torch.load(
                    weights_path, 
                    map_location=device,
                    weights_only=False
                )
            
            model = StrongGenerator().to(device)
            model.load_state_dict(checkpoint['generator'])
            model.eval()
            st.success("✅ Модель загружена через safe_globals")
            return model, device
            
        except Exception as e2:
            st.error(f"И этот способ не сработал: {e2}")
            return None, None
