#!/bin/bash
# Quick deployment verification script

echo "🔍 Pre-Deployment Checklist"
echo "=" | head -c 50; echo

# Check if all required files exist
echo "✓ Checking required files..."
files=("requirements.txt" "Procfile" "render.yaml" "start_render.sh" "src/app.py" "run.py")
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file exists"
    else
        echo "  ❌ $file MISSING!"
        exit 1
    fi
done

# Make start script executable
echo ""
echo "✓ Making start_render.sh executable..."
chmod +x start_render.sh
echo "  ✅ Done"

# Test if app can import without errors
echo ""
echo "✓ Testing Python imports..."
python -c "import sys; sys.path.insert(0, 'src'); from app import app; print('  ✅ App imports successfully')" || {
    echo "  ❌ Import failed! Check dependencies"
    exit 1
}

echo ""
echo "=" | head -c 50; echo
echo "🎉 All checks passed! Ready to deploy to Render"
echo ""
echo "Next steps:"
echo "1. Commit and push to GitHub"
echo "2. Go to https://dashboard.render.com"
echo "3. Click 'New +' → 'Web Service'"
echo "4. Connect your repository"
echo "5. Use these settings:"
echo "   - Build Command: pip install --upgrade pip && pip install -r requirements.txt"
echo "   - Start Command: bash start_render.sh"
echo "   - Health Check Path: /health"
echo ""
