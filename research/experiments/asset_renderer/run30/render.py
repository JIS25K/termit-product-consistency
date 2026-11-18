from pathlib import Path
import bpy,math,json,sys,time,hashlib,datetime
from mathutils import Vector,Matrix
R=Path(__file__).resolve().parent;O=R/'outputs';mode=sys.argv[sys.argv.index('--')+1] if '--' in sys.argv else 'inspect'
t=time.perf_counter();bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene;scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=24 if mode=='inspect' else 48;scene.cycles.use_denoising=True
scene.render.resolution_x=1024;scene.render.resolution_y=768;scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGBA';scene.render.film_transparent=True
scene.view_settings.view_transform='Standard';scene.view_settings.look='Medium High Contrast' if 'Medium High Contrast' in [] else 'None';scene.view_settings.exposure=0;scene.view_settings.gamma=1
scene.world=bpy.data.worlds.new('world');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs['Color'].default_value=(.8,.85,1,1);scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value=.3
meshpath=R/'data/berkeley/035_power_drill/poisson/textured.obj'
# Source OBJ has out-of-range normal references and no usemtl directive.
# Read positions/faces/UVs exactly; derive shading normals from the supplied faces.
verts=[];uvs=[];faces=[];faceuv=[]
for line in meshpath.read_text().splitlines():
 a=line.split()
 if not a:continue
 if a[0]=='v':verts.append(tuple(map(float,a[1:4])))
 elif a[0]=='vt':uvs.append(tuple(map(float,a[1:3])))
 elif a[0]=='f':
  refs=[x.split('/') for x in a[1:]];faces.append([int(x[0])-1 for x in refs]);faceuv.append([int(x[1])-1 for x in refs])
mesh=bpy.data.meshes.new('unchanged source geometry');mesh.from_pydata(verts,[],faces);mesh.update()
uvlayer=mesh.uv_layers.new(name='source UV')
for poly,indices in zip(mesh.polygons,faceuv):
 for loop,ui in zip(poly.loop_indices,indices):uvlayer.data[loop].uv=uvs[ui]
ob=bpy.data.objects.new('Berkeley drill',mesh);scene.collection.objects.link(ob);products=[ob]
material=bpy.data.materials.new('source texture');material.use_nodes=True
tex=material.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(meshpath.with_name('textured.png')))
material.node_tree.links.new(tex.outputs['Color'],material.node_tree.nodes['Principled BSDF'].inputs['Base Color']);mesh.materials.append(material)
assert len(mesh.polygons)==28592

# OBJ axis convention is explicit: no implicit upright reorientation by importer.
for ob in products:
 ob.matrix_world=Matrix.Identity(4)
bpy.context.view_layer.update();pts=[ob.matrix_world@Vector(c) for ob in products for c in ob.bound_box]
lo=Vector([min(p[i] for p in pts) for i in range(3)]);hi=Vector([max(p[i] for p in pts) for i in range(3)]);shift=Vector([-(lo.x+hi.x)/2,-(lo.y+hi.y)/2,-lo.z])
for ob in products:
 ob.location+=shift
 for poly in ob.data.polygons:poly.use_smooth=True
 for mat in ob.data.materials:
  if not mat or not mat.use_nodes:continue
  principled=next((n for n in mat.node_tree.nodes if n.type=='BSDF_PRINCIPLED'),None)
  if principled:
   principled.inputs['Roughness'].default_value=.5
   principled.inputs['Specular IOR Level'].default_value=.28
height=hi.z-lo.z
def light(name,loc,power,size,color):
 data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size;data.color=color
 ob=bpy.data.objects.new(name,data);scene.collection.objects.link(ob);ob.location=loc;ob.rotation_euler=(Vector((0,0,height*.5))-ob.location).to_track_quat('-Z','Y').to_euler()
light('large soft key',(-.35,-.3,.5),22,.45,(1,.92,.83));light('fill',(.3,.2,.3),8,.35,(.8,.88,1))
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.00015));floor=bpy.context.object;floor.name='continuous worktop'
mat=bpy.data.materials.new('neutral floor' if mode=='inspect' else 'matte oak worktop');mat.use_nodes=True;bs=mat.node_tree.nodes.get('Principled BSDF');bs.inputs['Roughness'].default_value=.68
bs.inputs['Base Color'].default_value=(.65,.65,.65,1) if mode=='inspect' else (.36,.235,.12,1)
if mode!='inspect':
 n=mat.node_tree.nodes;links=mat.node_tree.links;tc=n.new('ShaderNodeTexCoord');mapping=n.new('ShaderNodeVectorMath');mapping.operation='MULTIPLY';mapping.inputs[1].default_value=(6,150,8);links.new(tc.outputs['Position'] if 'Position' in tc.outputs else tc.outputs['Object'],mapping.inputs[0]);noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=3;noise.inputs['Detail'].default_value=2;links.new(mapping.outputs[0],noise.inputs['Vector']);ramp=n.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(.24,.135,.055,1);ramp.color_ramp.elements[1].color=(.48,.32,.17,1);links.new(noise.outputs['Fac'],ramp.inputs[0]);links.new(ramp.outputs['Color'],bs.inputs['Base Color'])
floor.data.materials.append(mat)
camera_data=bpy.data.cameras.new('camera');camera=bpy.data.objects.new('camera',camera_data);scene.collection.objects.link(camera);scene.camera=camera;camera_data.type='ORTHO';camera_data.ortho_scale=.31
records=[]
for name,az in [('A',-60),('B',-120)]:
 radians=math.radians(az);target=Vector((0,0,height*.50));camera.location=target+Vector((.6*math.cos(radians),.6*math.sin(radians),.24));camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler()
 if mode!='inspect':
  # Camera-aligned physical desk with a far edge. Empty distant background only.
  u=Vector((math.cos(radians),math.sin(radians),0));v=Vector((-math.sin(radians),math.cos(radians),0))
  corners=[-v*.8-u*.32,v*.8-u*.32,v*.8+u*.7,-v*.8+u*.7]
  for index,co in zip(floor.data.polygons[0].vertices,corners):floor.data.vertices[index].co=co
  floor.data.update()
 scene.render.filepath=str(O/(('neutral_' if mode=='inspect' else 'ours_')+name+'.png'));bpy.ops.render.render(write_still=True)
 records.append({'view':name,'azimuth_degrees':az,'elevation_degrees':math.degrees(math.atan2(.24,.6)),'camera_matrix':list(map(list,camera.matrix_world)),'path':scene.render.filepath})
report={'mode':mode,'renderer':bpy.app.version_string,'render_seconds':time.perf_counter()-t,'asset_vertices':sum(len(o.data.vertices) for o in products),'asset_polygons':sum(len(o.data.polygons) for o in products),'upright_extent_m':[hi.x-lo.x,hi.y-lo.y,height],'object_shift':list(shift),'axis_transform':'Berkeley Z-up source: identity rotation, center XY and set minimum Z to 0; no mesh repair','records':records,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(O/(mode+'_render.json')).write_text(json.dumps(report,indent=2));bpy.ops.wm.save_as_mainfile(filepath=str(O/(mode+'.blend')));print(json.dumps(report),flush=True)
